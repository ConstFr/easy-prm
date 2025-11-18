import json
from tqdm import tqdm
from torch.utils.data import Dataset
from copy import deepcopy
from datasets import load_dataset
import re
import numpy as np


STEP_SEP = "\n\n\n\n"
QUES_SEP = "\n\n"


def tokenize_data(dataset, tokenizer, max_length=None, num_samples=500):
    tokenize_data = []

    for i in range(len(dataset)):
        data = dataset[i]

        if data['reasoning_answer'] is None:
            continue
        question = re.findall(r'Question:\s*(.*?)\s*Response:', data['input_text'], re.DOTALL)[-1]
        input_text = question + f' {QUES_SEP}' + f' {STEP_SEP}'.join(data['reasoning_steps']) + f' {STEP_SEP}' # solution steps are separated by ' \n\n\n\n'
        input_id = tokenizer(input_text, add_special_tokens=False)

        labels = [-100] * len(input_id['input_ids'])
        tokens = tokenizer.convert_ids_to_tokens(input_id['input_ids'])

        # Replace -100 with 1 where token == STEP_SEP
        for i, tok in enumerate(tokens):
            if tok == STEP_SEP:
                labels[i] = 1
        input_id['labels'] = labels
        input_id['accuracy'] = data['accuracy'][0] 

        tokenize_data.append(input_id)

    return tokenize_data[:num_samples]


class TokenizedPRREvalDataset(Dataset):
    '''
    Tokenized PRM dataset
    Currently just stores all data in a list
    '''
    def __init__(self,  
                 dataset_name, 
                 tokenizer, 
                 max_length=None,
                 num_samples=500,
              ):

        super(TokenizedPRREvalDataset, self).__init__()

        self.dataset = load_dataset(dataset_name, split="test").shuffle(seed=42)
        
        self.tokenized_data = tokenize_data(self.dataset,
            tokenizer=tokenizer, 
            max_length=max_length,
            num_samples=num_samples,
            )
        

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, i):
        return self.tokenized_data[i]
