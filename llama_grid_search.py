import argparse
import itertools
import os
from copy import deepcopy

import yaml
from easydict import EasyDict as edict

from transformers import TrainingArguments, Trainer

from utils import (
    get_model,
    get_tokenizer,
    get_datasets_llama,
    get_collate_func,
    get_compute_loss_func,
    get_compute_metrics_llama,
)

try:
    import wandb
except ImportError:
    wandb = None


SEARCH_SPACE = {
    "learning_rate": [3.0e-5, 1.0e-4],
    "weight_decay": [0.0, 0.01],
    "per_device_train_batch_size": [4, 8],
    "optimizers": {
        "adamw": {
            "adam_beta1": [0.9, 0.95],
            "adam_beta2": [0.99],
            "adam_epsilon": [1.0e-8],
        },
        "sgd": {
            "sgd_momentum": [0.9],
            "sgd_nesterov": [False, True],
        },
        "muon": {
            "muon_beta1": [0.9],
            "muon_beta2": [0.99],
            "muon_eps": [1.0e-8],
        },
    },
}


def iter_trials():
    learning_rates = SEARCH_SPACE["learning_rate"]
    weight_decays = SEARCH_SPACE["weight_decay"]
    batch_sizes = SEARCH_SPACE["per_device_train_batch_size"]

    for lr, wd, bs in itertools.product(learning_rates, weight_decays, batch_sizes):
        for opt_name, opt_space in SEARCH_SPACE["optimizers"].items():
            keys = list(opt_space.keys())
            value_lists = [opt_space[k] for k in keys]
            for values in itertools.product(*value_lists):
                trial = {
                    "learning_rate": lr,
                    "weight_decay": wd,
                    "per_device_train_batch_size": bs,
                    "optimizer": opt_name,
                }
                trial.update(dict(zip(keys, values)))
                yield trial


def build_training_args(base_args, trial):
    args = deepcopy(base_args)

    args["learning_rate"] = trial["learning_rate"]
    args["weight_decay"] = trial["weight_decay"]
    args["per_device_train_batch_size"] = trial["per_device_train_batch_size"]

    optimizer = trial["optimizer"]
    if optimizer == "adamw":
        args["optim"] = "adamw_torch"
        args["adam_beta1"] = trial["adam_beta1"]
        args["adam_beta2"] = trial["adam_beta2"]
        args["adam_epsilon"] = trial["adam_epsilon"]
        args.pop("optim_args", None)
    elif optimizer == "sgd":
        args["optim"] = "sgd"
        momentum = trial["sgd_momentum"]
        nesterov = str(trial["sgd_nesterov"]).lower()
        args["optim_args"] = f"momentum={momentum},nesterov={nesterov}"
        for k in ["adam_beta1", "adam_beta2", "adam_epsilon"]:
            args.pop(k, None)
    elif optimizer == "muon":
        args["optim"] = "muon"
        beta1 = trial["muon_beta1"]
        beta2 = trial["muon_beta2"]
        eps = trial["muon_eps"]
        args["optim_args"] = f"beta1={beta1},beta2={beta2},eps={eps}"
        for k in ["adam_beta1", "adam_beta2", "adam_epsilon"]:
            args.pop(k, None)
    else:
        raise ValueError(f"Unsupported optimizer in search space: {optimizer}")

    return args


def format_run_name(trial, idx, total):
    parts = [
        f"gs-{idx+1:03d}-of-{total:03d}",
        f"opt={trial['optimizer']}",
        f"lr={trial['learning_rate']}",
        f"wd={trial['weight_decay']}",
        f"bs={trial['per_device_train_batch_size']}",
    ]
    return "|".join(parts)


def run_single_trial(base_configs, base_output_dir, trial, idx, total):
    configs = edict(deepcopy(base_configs))
    trial_args = build_training_args(configs.training_args, trial)

    run_name = format_run_name(trial, idx, total)
    run_output_dir = os.path.join(base_output_dir, "grid_search", run_name.replace("|", "_"))
    trial_args["output_dir"] = run_output_dir
    trial_args["run_name"] = run_name

    configs.training_args = trial_args

    if "wandb_project" in configs:
        os.environ["WANDB_PROJECT"] = configs.wandb_project
        os.environ.setdefault("WANDB_WATCH", "false")

    model = get_model(configs)
    tokenizer = get_tokenizer(configs.model_id)
    train_dataset, eval_dataset = get_datasets_llama(configs, tokenizer)
    collate_fn = get_collate_func(tokenizer)
    compute_loss = get_compute_loss_func()
    compute_metrics = get_compute_metrics_llama()

    training_args = TrainingArguments(
        label_names=["labels", "accuracy"],
        **configs.training_args,
    )

    if wandb is not None and "wandb_project" in configs:
        wandb_run = wandb.init(
            project=configs.wandb_project,
            name=run_name,
            group="llama_prm_base_grid_search",
            config={
                "base_config": configs.get("base_config_path", ""),
                **trial,
            },
        )
    else:
        wandb_run = None

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        processing_class=tokenizer,
        compute_loss_func=compute_loss,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    prr_key = "eval_PRR" if "eval_PRR" in eval_metrics else "PRR"
    prr_value = float(eval_metrics.get(prr_key, float("nan")))

    if wandb_run is not None:
        wandb_run.summary["best_PRR"] = prr_value
        wandb_run.finish()

    return prr_value, eval_metrics


def main(config_path):
    with open(config_path) as stream:
        configs = edict(yaml.safe_load(stream))

    if configs.type != "llama":
        raise ValueError("llama_grid_search is designed for type 'llama' configs only.")

    base_output_dir = configs.training_args.get("output_dir", "./runs/llama_prm_base")
    configs.base_config_path = config_path

    trials = list(iter_trials())
    total = len(trials)

    best_prr = float("-inf")
    best_trial = None

    for idx, trial in enumerate(trials):
        prr_value, _ = run_single_trial(configs, base_output_dir, trial, idx, total)
        if prr_value > best_prr:
            best_prr = prr_value
            best_trial = deepcopy(trial)

    if best_trial is not None:
        print("Best PRR:", best_prr)
        print("Best hyperparameters:")
        for k, v in best_trial.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search for llama PRM base config.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="train_configs/llama_prm_base.yml",
        help="Path to base llama config YAML.",
    )
    args = parser.parse_args()
    main(args.config)

