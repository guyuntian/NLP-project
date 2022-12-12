import json
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--val_size', type=int, default=2000)
args = parser.parse_args()

with open('data/train_data_20220406_10w.txt', encoding='utf-8') as f:
    js = f.read()
    js_split = js.split('\n')
    data = []
    for j in js_split:
        if len(j) > 2:
            data.append(eval(j))

random.seed(2022)
random.shuffle(data)

with open("data/my_train.txt", "w+") as fw:
    for i in range(100000-args.val_size):
        fw.write(str(data[i]) + '\n')
        
with open("data/my_val.txt", "w+") as fw:
    for i in range(args.val_size):
        fw.write(str(data[100000 - args.val_size + i]) + '\n')