import torch
import torch.nn as nn
import numpy as np
from tqdm import *
from data import getWord2Id, getDataLoader, cheat
from model import ModelConfig, TextCNN, MLP, LSTM
from torch.optim.lr_scheduler import *
import argparse
import math
import os

config = ModelConfig()
word2id = getWord2Id()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument('-p', '--pipeline', type=bool, default=False)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-l', '--lr', type=float, default=0.01)
    parser.add_argument('-s', '--seq_length', type=int, default=50)
    parser.add_argument('-m', '--model', default='TextCNN', choices=['TextCNN', 'MLP', 'LSTM'])
    parser.add_argument('-c', '--cheating', type=bool, default=False)
    parser.add_argument('-bi', '--bidirectional', type=bool, default=False)
    args = parser.parse_args()
    pipeline = args.pipeline
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    seq_length = args.seq_length
    cheating = args.cheating
    config.bidirectional = args.bidirectional
    config.num_directions = 2 if config.bidirectional else 1
    config.seq_len = seq_length
    print(args.model)
    if args.model == 'TextCNN':
        model = TextCNN(config=config).to(DEVICE)
    elif args.model == 'MLP':
        model = MLP(config=config).to(DEVICE)
    elif args.model == 'LSTM':
        model = LSTM(config=config).to(DEVICE)
    return pipeline, epochs, batch_size, lr, seq_length, cheating, model

def train(model, dataloader, criterion, optimizer, schduler, pipeline=False):
    model.train() # 训练模式
    running_loss = 0.0 # 统计损失
    for batch_idx, (x, y) in (tqdm(enumerate(dataloader, 0)) if not pipeline else enumerate(dataloader, 0)):
        optimizer.zero_grad() # 梯度清零
        outputs = model(x) # 前向传播，得到预测值
        loss = criterion(outputs, y) # 计算损失函数
        loss.backward() # 反向传播，计算梯度
        optimizer.step() # 更新参数
        running_loss += loss.item() # 累加损失
        if batch_idx % 100 == 99: # 训练一定batch之后打印提示信息
            print(f"[batch {batch_idx}] loss: {running_loss / 100}")
            running_loss = 0.0
    schduler.step() # 更新学习率
            
def validate(model, dataloader, criterion, cur_loss=math.inf, pipeline=False):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (x, y) in (tqdm(enumerate(dataloader, 0)) if not pipeline else enumerate(dataloader, 0)):
            outputs = model(x)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            if batch_idx % 50 == 49:
                val_loss /= 50
                print(f"[val batch {batch_idx}] loss: {val_loss}")
                if val_loss < cur_loss:
                    print(f"Validation loss decreased ({cur_loss:.6f} --> {val_loss:.6f}).  Saving model ...")
                    torch.save(model.state_dict(), f"./models/{model.__name__}_model.pth")
                    cur_loss = val_loss
                val_loss = 0.0
    return cur_loss
    
def test(model, dataloader, pipeline=False):
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for (x, y) in (tqdm(dataloader) if not pipeline else dataloader):
            outputs = model(x)
            _, predicted = torch.max(outputs.data, dim=1)
            y = y.bool()
            predicted = predicted.bool()
            tp += torch.bitwise_and(predicted, y).sum().item() # 被选中的并且选对了
            fp += torch.bitwise_and(predicted, torch.bitwise_not(y)).sum().item() # 被选中的但是选错了
            fn += torch.bitwise_and(torch.bitwise_not(predicted), y).sum().item() # 未被选中但是应该被选
            tn += torch.bitwise_and(torch.bitwise_not(predicted), torch.bitwise_not(y)).sum().item() # 未被选中并且确实不应该被选
    accuracy = (tp + tn) / (tp + fp + tn + fn) # 准确率：正确的操作有多少
    precision = tp / (tp + fp) # 精度：选的里面有多少是正确的
    recall = tp / (tp + fn) # 召回率：正确的里面有多少被选了
    fscore = (2 * precision * recall) / (precision + recall) # F分数
    print(f"Accuracy on test set: {100 * accuracy}%%")
    print(f"Precision on test set: {100 * precision}%%")
    print(f"Recall on test set: {100 * recall}%%")
    print(f"F-score on test set: {100 * fscore}%%")
    

if __name__ == '__main__':
    print("Getting args and Building model")
    pipeline, epochs, batch_size, lr, seq_len, cheating, model = parse_args()
    print("Finish getting args and Building model")
    print("Getting data loaders")
    train_loader, validation_loader, test_loader = getDataLoader(seq_len=seq_len, batch_size=batch_size)
    print("Finish getting data loaders")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    schduler = StepLR(optimizer, step_size=5, gamma=0.1)
    print("Start training")
    if not cheating:
        val_loss = math.inf
        if not os.path.exists("./models"):
            os.mkdir("./models")
        for epoch in range(1, epochs + 1):
            print(f"Begin [epoch {epoch}]")
            print("Begin training")
            train(model, train_loader, criterion, optimizer, schduler, pipeline=pipeline)
            print("Begin validation")
            val_loss = validate(model, validation_loader, criterion, cur_loss=val_loss, pipeline=pipeline)
        if os.path.exists(f"./models/{model.__name__}_model.pth"):
            print(f"Loading {model.__name__}_model")
            model.load_state_dict(torch.load(f"./models/{model.__name__}_model.pth"))
            print("Final test")
            test(model, test_loader, pipeline=pipeline)
        else:
            pass
    else:
        cheat_loader = cheat(seq_len=seq_len, batch_size=batch_size)
        for epoch in range(1, epochs + 1):
            print(f"Begin [epoch {epoch}]")
            train(model, cheat_loader, criterion, optimizer, schduler)
            test(model, test_loader)
        