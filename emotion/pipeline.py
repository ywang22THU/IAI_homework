import os
import argparse

lrs = [0.01, 0.001, 0.0001, 0.00001]
batch_sizes = [16, 32, 64, 128]

if not os.path.exists("logs"):
    os.makedirs("logs")

def TextCNN_pipeline():
    for lr in lrs:
        for batch_size in batch_sizes:
            print(f"Running TextCNN experiment with lr = {lr} and batch_size = {batch_size}")
            log_file = open(f"logs/TextCNN_log_{lr}_{batch_size}.txt", "w")
            log_file.truncate(0)
            os.system(f"python train.py -l {lr} -b {batch_size} -p 1 >> logs/TextCNN_log_{lr}_{batch_size}.txt")

def MLP_pipeline():
    for lr in lrs:
        for batch_size in batch_sizes:
            print(f"Running MLP experiment with lr = {lr} and batch_size = {batch_size}")
            log_file = open(f"logs/MLP_log_{lr}_{batch_size}.txt", "w")
            log_file.truncate(0)
            os.system(f"python train.py -l {lr} -b {batch_size} -m MLP -p 1 >> logs/MLP_log_{lr}_{batch_size}.txt")
            
def LSTM_pipeline():
    for lr in lrs:
        for batch_size in batch_sizes:
            print(f"Running LSTM experiment with lr = {lr} and batch_size = {batch_size}")
            log_file = open(f"logs/LSTM_log_{lr}_{batch_size}.txt", "w")
            log_file.truncate(0)
            os.system(f"python train.py -l {lr} -b {batch_size} -m LSTM -p 1 >> logs/LSTM_log_{lr}_{batch_size}.txt")

def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument("-m", "--model", type=str, default="TextCNN", choices=["TextCNN", "MLP", "LSTM"])
    parser.add_argument("-a", "--all", action="store_true")
    args = parser.parse_args()
    return args.model, args.all

if __name__ == "__main__":
    model, all = parse_args()
    if all:
        TextCNN_pipeline()
        MLP_pipeline()
        LSTM_pipeline()
        exit()
    if model == "TextCNN":
        TextCNN_pipeline()
    elif model == "MLP":
        MLP_pipeline()
    elif model == "LSTM":
        LSTM_pipeline()