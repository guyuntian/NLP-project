from transformers import AutoTokenizer, AutoModel, set_seed, get_scheduler
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from accelerate import Accelerator
import os
import random
import copy
os.environ["TOKENIZERS_PARALLELISM"] = "true"

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument("--output_dir", type=str, default=None) 
parser.add_argument("--model_name_or_path", type=str, default=None) 
parser.add_argument("--head_path", type=str, default=None)
parser.add_argument("--debug_val_size", type=int, default=None)
parser.add_argument("--threshold1", type=float, default=0.1)
parser.add_argument("--threshold2", type=float, default=0.7)

args = parser.parse_args()

set_seed(args.seed)

import json
Labels = dict()
with open("data/case_classification.txt", encoding='utf-8') as f:
    js = f.read()
    js_split = js.split()
    cur = [0,0,0,0]
    for i in range(len(js_split) // 2):
        cur[int(js_split[2*i+1])] = js_split[2*i]
        Labels[js_split[2*i]] = cur[int(js_split[2*i+1]) - 1]
        
father_to_son = dict()
father_to_son['root'] = []
for label in list(Labels.keys()):
    father_to_son[label] = []
for label in list(Labels.keys()):
    father_to_son[Labels[label]].append(label) if Labels[label] else father_to_son['root'].append(label)

class classification_Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(768, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = nn.Sigmoid()(x)
        return x.unsqueeze(-1)
    
def get_choice(model, head, data, choices):
    
    val_data = []
    for choice in choices:
        tmp = copy.deepcopy(data)
        tmp['aq'] = choice + '。' + tmp['aq']
        tmp = tokenizer(tmp['aq'], truncation=True, padding='max_length', max_length=args.max_length)
        val_data.append(tmp)

    dataset = Dataset.from_list(val_data)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    with torch.no_grad():
        inputs = dataset[:]
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        res = model(**inputs)
        logit = head(res.last_hidden_state[:,0])
        predict = choices[torch.argmax(logit)]
    return predict, torch.max(logit).item()

def predict(model, head):
    with open('data/test_data_2022_1w.txt', encoding='utf-8') as f:  #test_data_2022_1w.txt
        js = f.read()
        js_split = js.split('\n')
        val_data = []
        for j in js_split:
            if len(j) > 2:
                data = eval(j)
                val_data.append(data)
                    
    if args.debug_val_size is not None:
        val_data = val_data[:args.debug_val_size]
    model.eval()
    predict = []
    predict_level = []
    for data in tqdm(val_data):
        choices = father_to_son['root']
        choice, _ = get_choice(model, head, data, choices)
        if len(father_to_son[choice]) == 0:
            predict.append(choice)
            predict_level.append(1)
            continue
        choices = father_to_son[choice]
        choice2, score = get_choice(model, head, data, choices)
        if score < args.threshold1:
            predict.append(choice)
            predict_level.append(1)
            continue
        choice = choice2

        if len(father_to_son[choice]) == 0:
            predict.append(choice)
            predict_level.append(2)
            continue
        choice2, score = get_choice(model, head, data, choices)
        choices = father_to_son[choice]
        if score < args.threshold2:
            predict.append(choice)
            predict_level.append(2)
            continue
        predict.append(choice2)
        predict_level.append(3)
    
    with open(args.output_dir, 'w+', encoding='utf-8') as f:
        for i in range(len(predict)):
            f.write(str(predict_level[i]))
            f.write(' ')
            f.write(predict[i])
            f.write('\n')
    
            
            
accelerator = Accelerator() #accelerator.num_processes = 1
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModel.from_pretrained(args.model_name_or_path)
head = classification_Head() if args.head_path is None else torch.load(args.head_path)
head = head.cuda()
model = accelerator.prepare(model)
predict(model, head)