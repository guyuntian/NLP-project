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
import numpy as np
from torch.utils.tensorboard import SummaryWriter

os.environ["TOKENIZERS_PARALLELISM"] = "true"

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--neg_sample", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--head_lr", type=float, default=3e-4)
parser.add_argument("--warmup_steps", type=int, default=0)
parser.add_argument("--training_steps", type=int, default=10)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--model_name_or_path", type=str, default="bert-base-chinese") #"bert-base-chinese"
parser.add_argument("--head_path", type=str, default=None)
parser.add_argument("--debug_train_size", type=int, default=None)
parser.add_argument("--debug_val_size", type=int, default=None)

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
    
os.makedirs(args.output_dir, exist_ok=True)
log_writer = SummaryWriter(log_dir=args.output_dir)


def get_dataloader(tokenizer):
    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(examples['aq'], truncation=True, padding='max_length', max_length=args.max_length)
        return result
    
    def append_data(datas, data):
        name = []
        name.append(data['ay'])
        while name[-1]:
            new_data = copy.deepcopy(data)
            new_data['aq'] = name[-1] + "。" + new_data['aq']
            new_data['ay'] = 1
            datas.append(new_data)
            name.append(Labels[name[-1]])
        name[-1] = 'root'
        choose_set = [father_to_son[item_name] for item_name in name[1:]] if len(name) == 4 else [father_to_son[item_name] for item_name in name]
        for tmp_list in choose_set:
            if len(tmp_list) == 0:
                continue
            tmp_list = random.sample(tmp_list, args.neg_sample) if len(tmp_list) > args.neg_sample else tmp_list
            for label in tmp_list:
                if label in name:
                    continue
                new_data = copy.deepcopy(data)
                new_data['aq'] = label + "。" + new_data['aq']
                new_data['ay'] = 0
                datas.append(new_data)

    with open('data/my_train.txt', encoding='utf-8') as f:
        js = f.read()
        js_split = js.split('\n')
        train_data = []
        for j in js_split:
            if len(j) > 2:
                data = eval(j)
                data['aq'] = data['aq'][:512]
                append_data(train_data, data)
    if args.debug_train_size is not None:
        train_data = train_data[:args.debug_train_size]
    train_dataset = Dataset.from_list(train_data)
    print(train_dataset)
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    print("map finish!")
    train_dataset = train_dataset.rename_column("ay", "labels")
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=8)
    return train_loader

class classification_Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(768, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = nn.Sigmoid()(x)
        return x.unsqueeze(-1)
    
def set_optimizer_scheduler(model, head):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.learning_rate,
        },
        {
            "params": [p for n, p in head.named_parameters()],
            "weight_decay": args.weight_decay,
            "lr": args.head_lr,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
#     for n, p in model.named_parameters():
#         p.requires_grad = False
            
    scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.training_steps)
    return optimizer, scheduler

def get_score(answer, predict):
    score = 0
    for i in range(len(answer)):
        level = 1
        a = answer[i]
        while Labels[a]:
            level += 1
            a = Labels[a]
        
        if level == 1:
            if answer[i] == predict[i]:
                score += 1
        elif level == 2:
            if answer[i] == predict[i]:
                score += 1
            elif Labels[answer[i]] == predict[i]:
                score += 0.5
        else:
            if answer[i] == predict[i]:
                score += 1
            elif Labels[answer[i]] == predict[i]:
                score += 0.8
            elif Labels[Labels[answer[i]]] == predict[i]:
                score += 0.5
    return score / len(answer)
    
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

def evaluate(model, head):
    with open('data/my_val.txt', encoding='utf-8') as f:
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
    Loss = 0
    answer, predict1, predict2, predict3 = [], [], [], []
    for data in val_data:
        answer.append(data['ay'])
        choices = father_to_son['root']
        choice, _ = get_choice(model, head, data, choices)
        predict1.append(choice)
        if len(father_to_son[choice]):
            choices = father_to_son[choice]
            choice, _ = get_choice(model, head, data, choices)
        predict2.append(choice)
        if len(father_to_son[choice]):
            choices = father_to_son[choice]
            choice, _ = get_choice(model, head, data, choices)
        predict3.append(choice)
    
    return get_score(answer, predict1), get_score(answer, predict2), get_score(answer, predict3)
    

def train(model, head, optimizer, scheduler, train_loader, accelerator, tokenizer):
    model.train()
    for epoch in range(args.training_steps):
        pbar = enumerate(tqdm(train_loader))
        for data_iter_step, inputs in pbar:
            labels = inputs['labels']
            del inputs['labels']
            res = model(**inputs)
            logit = head(res.last_hidden_state[:,0])
            loss = -torch.mean(labels * torch.log(logit) + (1-labels) * torch.log(1-logit))
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            
            epoch_1000x = int((data_iter_step / len(train_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss.item(), epoch_1000x)

        scheduler.step()
        
        first, second, third = evaluate(model, head)
        log_writer.add_scalar('perf/first_level', first, epoch)
        log_writer.add_scalar('perf/second_level', second, epoch)
        log_writer.add_scalar('perf/third_level', third, epoch)
        log_writer.flush()
        
        unwrapped_model = accelerator.unwrap_model(model)
        outputs_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
        unwrapped_model.save_pretrained(outputs_dir)
        tokenizer.save_pretrained(outputs_dir)

        output_dir = os.path.join(outputs_dir, "head.pth")
        torch.save(head, output_dir)
            
            
accelerator = Accelerator() #accelerator.num_processes = 1
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModel.from_pretrained(args.model_name_or_path)
train_loader = get_dataloader(tokenizer)
head = classification_Head() if args.head_path is None else torch.load(args.head_path)
head = head.cuda()
optimizer, scheduler = set_optimizer_scheduler(model, head)

model, optimizer, scheduler, train_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader)
train(model, head, optimizer, scheduler, train_loader, accelerator, tokenizer)