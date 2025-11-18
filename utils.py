from transformers import (
    AutoTokenizer, 
    DataCollatorForTokenClassification, 
    AutoModelForCausalLM,
    AutoModelForTokenClassification)
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from prm_datasets import TokenizedPRMDataset
from eval_datasets import TokenizedPRREvalDataset
import evaluate
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

import logging

logging.basicConfig(
    filename='app.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)



def get_random_scores(function, metrics, num_iter=1000, seed=42):
    np.random.seed(seed)
    rand_scores = np.arange(len(metrics))

    value = []
    for i in range(num_iter):
        np.random.shuffle(rand_scores)
        rand_val = function(rand_scores, metrics)
        value.append(rand_val)
    return np.mean(value)


def normalize(target):
    min_t, max_t = np.min(target), np.max(target)
    if np.isclose(min_t, max_t):
        min_t -= 1
        max_t += 1
    target = (np.array(target) - min_t) / (max_t - min_t)
    return target


class PredictionRejectionArea():
    """
    Calculates area under Prediction-Rejection curve.
    """

    def __init__(self, max_rejection: float = 1.0):
        """
        Parameters:
            max_rejection (float): a maximum proportion of instances that will be rejected.
                1.0 indicates entire set, 0.5 - half of the set
        """
        super().__init__()
        self.max_rejection = max_rejection

    def __str__(self):
        if self.max_rejection == 1:
            return "prr"
        return f"prr_{self.max_rejection}"

    def __call__(self, estimator, target) -> float:
        """
        Measures the area under the Prediction-Rejection curve between `estimator` and `target`.

        Parameters:
            estimator (List[int]): a batch of uncertainty estimations.
                Higher values indicate more uncertainty.
            target (List[int]): a batch of ground-truth uncertainty estimations.
                Higher values indicate less uncertainty.
        Returns:
            float: area under the Prediction-Rejection curve.
                Higher values indicate better uncertainty estimations.
        """
        target = normalize(target)
        # ue: greater is more uncertain
        ue = np.array(estimator)
        num_obs = len(ue)
        num_rej = int(self.max_rejection * num_obs)
        # Sort in ascending order: the least uncertain come first
        ue_argsort = np.argsort(ue)
        # want sorted_metrics to be increasing => smaller scores is better
        sorted_metrics = np.array(target)[ue_argsort]
        # Since we want all plots to coincide when all the data is discarded
        cumsum = np.cumsum(sorted_metrics)[-num_rej:]
        scores = (cumsum / np.arange((num_obs - num_rej) + 1, num_obs + 1))[::-1]
        prr_score = np.sum(scores) / num_rej
        return prr_score


def _delete_nans(ue, metric):
    metric = np.asarray(metric)

    # Clipping, because some evaluation metrics cannot work with nan ue scores.
    clipped_ue = np.nan_to_num(ue, nan=-1e7, neginf=-1e7, posinf=1e7)

    is_nan_metric_mask = np.isnan(metric)
    clipped_ue = clipped_ue[~is_nan_metric_mask]
    new_metric = metric[~is_nan_metric_mask]

    return clipped_ue, new_metric


def normalize_metric(target_score, oracle_score, random_score):
    if not (oracle_score == random_score):
        target_score = (target_score - random_score) / (oracle_score - random_score)
    return target_score


def calculate_prr_05_normalized(generation_metric, estimator_values):
    for ue_metric in [PredictionRejectionArea(max_rejection=0.5)]:
        oracle_score_all = ue_metric(
            -np.array(generation_metric), np.array(generation_metric)
        )
        random_score_all = get_random_scores(
            ue_metric, np.array(generation_metric)
        )
        ue, metric = _delete_nans(estimator_values, generation_metric)
        ue_metric_val = ue_metric(ue, metric)
        ue_metric_val_normalized = normalize_metric(ue_metric_val, oracle_score_all, random_score_all)
    return ue_metric_val_normalized


def get_model(configs):
    

    model = AutoModelForCausalLM.from_pretrained(configs.model_id, device_map='auto')


    if 'lora_config' in configs:
        print('Using LoRA')
        lora_config = LoraConfig(**configs.lora_config)
        model = get_peft_model(model, lora_config)
        
    return model


def get_model_bert(configs, tokenizer):

    # Token classification is binary (+ / -), so we need two logits.
    model = AutoModelForTokenClassification.from_pretrained(
        configs.model_id,
        num_labels=2,
        # device_map='auto',
    )
    model.resize_token_embeddings(len(tokenizer))

    return model


def get_tokenizer(model_id):
        
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token #llama doesn't define pad token, so we need to do this
    tokenizer.padding_side='right' # we need to pad from right (so that we can do eval mask id trick for eval)


    return tokenizer


EXTRA_SPECIALS = ["[QUES_SEP]", "[STEP_SEP]"]

def get_tokenizer_bert(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    tokenizer.add_special_tokens({"additional_special_tokens": EXTRA_SPECIALS})
    return tokenizer


# def get_tokenizer_bert(model_id):
        
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     tokenizer.padding_side='right' # we need to pad from right (so that we can do eval mask id trick for eval)


#     return tokenizer


def get_datasets(configs, tokenizer, model_type="llama"):
    
    t_dataset = TokenizedPRMDataset(configs.train_data_path, 
                                    tokenizer,
                                    model_type=model_type,
                                    label_last_n = configs.train_label_last_n if 'train_label_last_n' in configs else None,
                                    max_length=configs.max_length if 'max_length' in configs else None,
                                    use_augs=configs.use_augs if 'use_augs' in configs else True)
    e_dataset = TokenizedPRMDataset(configs.eval_data_path, 
                                    tokenizer,
                                    label_last_n = configs.eval_label_last_n if 'eval_label_last_n' in configs else None,
                                    max_length=configs.max_length if 'max_length' in configs else None,
                                    use_augs=configs.use_augs if 'use_augs' in configs else True) if configs.eval_data_path is not None else None
    return t_dataset, e_dataset


def get_datasets_llama(configs, tokenizer, model_type="llama"):
    
    t_dataset = TokenizedPRMDataset(
        configs.train_data_path, 
        tokenizer,
        model_type=model_type,
        label_last_n = configs.train_label_last_n if 'train_label_last_n' in configs else None,
        max_length=configs.max_length if 'max_length' in configs else None,
        use_augs=configs.use_augs if 'use_augs' in configs else True
    )
    e_dataset = TokenizedPRREvalDataset(
        configs.eval_data_path, 
        tokenizer,
        model_type=model_type,
        max_length=configs.max_length if 'max_length' in configs else None,
        num_samples=configs.eval_num_samples if 'eval_num_samples' in configs else 500
    )
    return t_dataset, e_dataset


def get_collate_func(tokenizer):
      
    return DataCollatorForTokenClassification(tokenizer=tokenizer, 
                                                        padding='longest', 
                                                        label_pad_token_id=-100,
                                                        return_tensors='pt')


def get_compute_loss_func():
      
    def compute_loss_func(outputs, labels, num_items_in_batch):

        # output logits are in shape (B, L, V) - batch, seq length, vocab size

        # 12 is ID of '-', 10 is ID of '+' (for both Llama and Qwen tokenizer)
        # TODO: change so its more flexible for different tokenizers
        logits = outputs.logits[:,:,[12,10]].reshape(-1,2)


        # for eval, num_items_in_batch is None
        if num_items_in_batch is None:
            loss = F.cross_entropy(input=logits,
                            target=labels.flatten(),
                            ignore_index=-100)
            return loss

        # num_items_in_batch
        # https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/trainer.py#L5142
        loss = F.cross_entropy(input=logits,
                            target=labels.flatten(),
                            ignore_index=-100,
                            reduction='sum')

        return loss / num_items_in_batch
    
    return compute_loss_func


def get_compute_loss_func_bert():
      
    def compute_loss_func(outputs, labels, num_items_in_batch):

        # output logits are in shape (B, L, 2) - batch, seq length, num_classes
        # TODO: change so its more flexible for different tokenizers
        logits = outputs.logits.reshape(-1, outputs.logits.shape[-1])  # [B*L, 2]
        # logging.info(f"{logits.shape=}")
        # logging.info(f"{labels.flatten().shape}")


        # for eval, num_items_in_batch is None
        if num_items_in_batch is None:
            loss = F.cross_entropy(input=logits,
                            target=labels.flatten(),
                            ignore_index=-100)
            return loss

        # num_items_in_batch
        # https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/trainer.py#L5142
        loss = F.cross_entropy(input=logits,
                            target=labels.flatten(),
                            ignore_index=-100,
                            reduction='sum')

        return loss / num_items_in_batch
    
    return compute_loss_func 


def get_compute_metrics():
    '''
    gets metrics for precision, recall, f1 score...
    '''
       
    
    accuracy = evaluate.load('accuracy')
    precision = evaluate.load('precision')
    recall = evaluate.load('recall')
    f1 = evaluate.load('f1')


    def compute_metrics(eval_pred):
        logits, labels = eval_pred


        label_mask_PRM = (labels!=-100)

        labels_PRM = labels[label_mask_PRM]
        logits_PRM = logits[:,:,[12, 10]][label_mask_PRM]

        pred_PRM = np.argmax(logits_PRM, axis=-1)
        predf_PRM = softmax(logits_PRM)[:,1]


        results = {
            'PRM Accuracy': accuracy.compute(predictions=pred_PRM, references=labels_PRM)['accuracy'],
            'PRM Precision': precision.compute(predictions=pred_PRM, references=labels_PRM, zero_division=0.0)['precision'],
            'PRM Recall': recall.compute(predictions=pred_PRM, references=labels_PRM)['recall'],
            'PRM Specificty': recall.compute(predictions=pred_PRM, references=labels_PRM, pos_label=0)['recall'],
            'PRM NPV': precision.compute(predictions=pred_PRM, references=labels_PRM, pos_label= 0, zero_division=0.0)['precision'], # negative predictive value, unPrecision
            'PRM F1': f1.compute(predictions=pred_PRM, references=labels_PRM)['f1'],
            'PRM F1 Neg': f1.compute(predictions=pred_PRM, references=labels_PRM, pos_label=0)['f1'],
            'PRM F1 AUC': roc_auc_score(labels_PRM, predf_PRM),
            }
    

        return results
    
    return compute_metrics


def get_compute_metrics_llama():
    '''
    gets metrics
    '''
       

    def compute_metrics(eval_pred):
        logits, (labels, accuracy) = eval_pred

        mask = (labels!=-100)

        logits_PRM = logits[:,:,[12, 10]]
        scores = softmax(logits_PRM, axis=-1)[..., 1]

        # [0.99], [0.5], [0.76], [1.0], ...
        # 

        mask_f = mask.astype(np.float32)
        sum_scores = (scores * mask_f).sum(axis=1)
        counts = mask_f.sum(axis=1)

        mean_scores = np.divide(
            sum_scores,
            counts,
            out=np.zeros_like(sum_scores),  # fill zeros where counts==0
            where=counts > 0
        )
        step_probs = 1.0 - mean_scores

        # logging.info(f"{step_probs.shape=}")
        # logging.info(f"{accuracy.shape=}")
        # logging.info(f"{labels.shape=}")
        # logging.info(f"{logits.shape=}")
        # logging.info(f"{logits_PRM.shape=}")
        # logging.info(f"{scores.shape=}")
        # logging.info(f"{mask.shape=}")

        results = {
            'PRR': calculate_prr_05_normalized(accuracy, step_probs)
            }

        return results
    
    return compute_metrics



def get_compute_metrics_bert():
    '''
    gets metrics for precision, recall, f1 score...
    '''
       
    
    accuracy = evaluate.load('accuracy')
    precision = evaluate.load('precision')
    recall = evaluate.load('recall')
    f1 = evaluate.load('f1')


    def compute_metrics(eval_pred):
        logits, labels = eval_pred


        label_mask_PRM = (labels!=-100)

        labels_PRM = labels[label_mask_PRM]
        logits_PRM = logits[label_mask_PRM]

        pred_PRM = np.argmax(logits_PRM, axis=-1)
        predf_PRM = softmax(logits_PRM)[:,1]


        results = {
            'PRM Accuracy': accuracy.compute(predictions=pred_PRM, references=labels_PRM)['accuracy'],
            'PRM Precision': precision.compute(predictions=pred_PRM, references=labels_PRM, zero_division=0.0)['precision'],
            'PRM Recall': recall.compute(predictions=pred_PRM, references=labels_PRM)['recall'],
            'PRM Specificty': recall.compute(predictions=pred_PRM, references=labels_PRM, pos_label=0)['recall'],
            'PRM NPV': precision.compute(predictions=pred_PRM, references=labels_PRM, pos_label= 0, zero_division=0.0)['precision'], # negative predictive value, unPrecision
            'PRM F1': f1.compute(predictions=pred_PRM, references=labels_PRM)['f1'],
            'PRM F1 Neg': f1.compute(predictions=pred_PRM, references=labels_PRM, pos_label=0)['f1'],
            'PRM F1 AUC': roc_auc_score(labels_PRM, predf_PRM),
            }
    

        return results
    
    return compute_metrics
