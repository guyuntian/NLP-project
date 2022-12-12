from transformers import AutoTokenizer, AutoModelForMaskedLM, set_seed, get_scheduler, DataCollatorForLanguageModeling
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
from torch.utils.tensorboard import SummaryWriter
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--warmup_steps", type=int, default=0)
parser.add_argument("--training_steps", type=int, default=10)
parser.add_argument("--mlm_probability", type=float, default=0.15)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--model_name_or_path", type=str, default="bert-base-chinese")

args = parser.parse_args()

set_seed(args.seed)

def get_dataloader(tokenizer, data_collator):
    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(examples['aq'], truncation=True, padding='max_length', max_length=args.max_length)
        return result
    
    train_data = []
    with open('data/train_data_20220406_10w.txt', encoding='utf-8') as f:
        js = f.read()
        js_split = js.split('\n')
        for j in js_split:
            if len(j) > 2:
                data = eval(j)
                train_data.append(data)
#     with open('data/test_data_2022_1w.txt', encoding='utf-8') as f:
#         js = f.read()
#         js_split = js.split('\n')
#         for j in js_split:
#             if len(j) > 2:
#                 data = eval(j)
#                 train_data.append(data)
    train_dataset = Dataset.from_list(train_data)
    print(train_dataset)
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    print("map finish!")
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=True, drop_last=False, num_workers=8)
    return train_loader
    
def set_optimizer_scheduler(model):
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
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
            
    scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.training_steps)
    return optimizer, scheduler
    

def train(model, optimizer, scheduler, train_loader, accelerator):
    model.train()
    for epoch in range(args.training_steps):
        pbar = enumerate(tqdm(train_loader))
        for data_iter_step, inputs in pbar:
            res = model(**inputs)
            loss = res.loss
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            epoch_1000x = int((data_iter_step / len(train_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss.item(), epoch_1000x)
            
        log_writer.flush()
        scheduler.step()
    if args.output_dir is not None:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        
log_writer = SummaryWriter(log_dir=args.output_dir)
accelerator = Accelerator() #accelerator.num_processes = 1
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)
train_loader = get_dataloader(tokenizer, data_collator)
model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
optimizer, scheduler = set_optimizer_scheduler(model)
model, optimizer, scheduler, train_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader)
train(model, optimizer, scheduler, train_loader, accelerator)