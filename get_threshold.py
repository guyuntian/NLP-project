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
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

os.environ["TOKENIZERS_PARALLELISM"] = "true"

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument("--model_name_or_path", type=str, default="bert-base-chinese") #"bert-base-chinese"
parser.add_argument("--head_path", type=str, default=None)
parser.add_argument("--debug_val_size", type=int, default=None)
parser.add_argument("--start1", type=float, default=0.1)
parser.add_argument("--end1", type=float, default=0.2)
parser.add_argument("--start2", type=float, default=0.1)
parser.add_argument("--end2", type=float, default=0.2)
parser.add_argument("--num", type=int, default=100)


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

def get_prediction(model, head):
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
    answer, predict1, predict2, predict3, score1, score2 = [], [], [], [], [], []
    for data in tqdm(val_data):
        answer.append(data['ay'])
        choices = father_to_son['root']
        choice, _ = get_choice(model, head, data, choices)
        predict1.append(choice)
        score = 0
        if len(father_to_son[choice]):
            choices = father_to_son[choice]
            choice, score = get_choice(model, head, data, choices)
        predict2.append(choice)
        score1.append(score)
        if len(father_to_son[choice]):
            choices = father_to_son[choice]
            choice, score = get_choice(model, head, data, choices)
        predict3.append(choice)
        score2.append(score)
    
    return answer, predict1, predict2, predict3, score1, score2
    
def score_for_predict(answer, predict1, predict2, predict3, score1, score2, threshold1, threshold2):
    predict = []
    for i in range(len(answer)):
        if score1[i] < threshold1:
            predict.append(predict1[i])
        elif score2[i] < threshold2:
            predict.append(predict2[i])
        else:
            predict.append(predict3[i])
    return get_score(answer, predict)
    
def get_best_threshold(model, head):
    num = args.num
    answer, predict1, predict2, predict3, score1, score2 = get_prediction(model, head)
    progress_bar = tqdm(range(num * num))
    best_score, best_i, best_j = -1, 0, 0
    score_all = np.zeros((num, num))
    I = np.arange(args.start1, args.end1, (args.end1 - args.start1) / num)
    J = np.arange(args.start2, args.end2, (args.end2 - args.start2) / num)
    for i in range(num):
        for j in range(num):
            score = score_for_predict(answer, predict1, predict2, predict3, score1, score2, I[i], J[j])
            if score > best_score:
                best_score = score
                best_i = i
                best_j = j
            score_all[i, j] = score
            progress_bar.update(1)
            
            
    fig = plt.figure()  #定义新的三维坐标轴
    ax3 = plt.axes(projection='3d')

    #定义三维数据
    xx = np.arange(0,num)
    yy = np.arange(0,num)
    X, Y = np.meshgrid(xx, yy)
    Z = score_all[X, Y]


    #作图
    ax3.plot_surface(I[X],J[Y],Z,cmap='rainbow')
    #ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
    plt.show()
    plt.savefig("outputs/fig.png")

            
    print("best score: ", best_score, "best threshold:", (I[best_i], J[best_j]))
    


            
accelerator = Accelerator() #accelerator.num_processes = 1
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModel.from_pretrained(args.model_name_or_path)
head = classification_Head() if args.head_path is None else torch.load(args.head_path)
head = head.cuda()
model = accelerator.prepare(model)
get_best_threshold(model, head)

    
#     with open('data/my_val.txt', encoding='utf-8') as f:
#         js = f.read()
#         js_split = js.split('\n')
#         val_data = []
#         for j in js_split:
#             if len(j) > 2:
#                 data = eval(j)
#                 val_data.append(data)
                    
#     if args.debug_val_size is not None:
#         val_data = val_data[:args.debug_val_size]
#     model.eval()
#     Loss = 0
#     thresholds1 = [args.threshold1_start, (args.threshold1_start + args.threshold1_end) / 2, args.threshold1_end]
#     thresholds2 = [args.threshold2_start, (args.threshold2_start + args.threshold2_end) / 2, args.threshold2_end]
#     scores = np.zeros((len(thresholds1), len(thresholds2)))
#     for i in tqdm(range(len(thresholds1))):
#         for j in range(len(thresholds2)):
#             answer, predict = [], []
#             for data in val_data:
#                 answer.append(data['ay'])
#                 choices = father_to_son['root']
#                 choice, _ = get_choice(model, head, data, choices)
#                 if len(father_to_son[choice]) == 0:
#                     predict.append(choice)
#                     continue
#                 choices = father_to_son[choice]
#                 choice2, score = get_choice(model, head, data, choices)
#                 if score < thresholds1[i]:
#                     predict.append(choice)
#                     continue
#                 choice = choice2
                
#                 if len(father_to_son[choice]) == 0:
#                     predict.append(choice)
#                     continue
#                 choices = father_to_son[choice]
#                 choice2, score = get_choice(model, head, data, choices)
#                 if score < thresholds2[j]:
#                     predict.append(choice)
#                     continue
#                 predict.append(choice2)
                
#             scores[i, j] = get_score(answer, predict)
            
#     max_score = np.max(scores)
#     for i in range(len(thresholds1)):
#         for j in range(len(thresholds2)):
#             if scores[i, j] == max_score:
#                 return thresholds1[i], thresholds2[j], max_score