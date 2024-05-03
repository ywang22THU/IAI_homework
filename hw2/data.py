import gensim
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import *

train_data = "./Dataset/train.txt"
validation_data = "./Dataset/validation.txt"
test_data = "./Dataset/test.txt"
root = "./Dataset/"

def getWord2Id():
    word2id = {}
    with open(train_data, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            words = line.strip().split()[1:]
            for word in words:
                if word not in word2id:
                    word2id[word] = len(word2id) + 1
    return word2id

def getWord2Vec(word2id):
    preModel = gensim.models.KeyedVectors.load_word2vec_format('./Dataset/wiki_word2vec_50.bin', binary=True)
    word2vec = np.array(np.zeros([len(word2id) + 1, preModel.vector_size]), dtype=np.float32)
    for word in word2id:
        try:
            word2vec[word2id[word]] = preModel[word]
        except Exception:
            pass
    return word2vec

def readInput(file_path, word2id: dict, seq_len=50):
    contents, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            sentence = line.strip().split()[1:]
            label = line.strip().split()[0]
            content = []
            for word in sentence:
                content.append(word2id.get(word, 0))
            # 统一长度
            content = content[:seq_len] + [0] * (seq_len - len(content))
            labels.append(int(label))
            contents.append(content)
    return np.array(contents), np.array(labels)


class TextDataset(Dataset):
    def __init__(self, file_path, word2id: dict, seq_len=50):
        contents, labels = readInput(file_path, word2id, seq_len)
        super(TextDataset, self).__init__()
        self.contents = torch.LongTensor(contents)
        self.labels = torch.LongTensor(labels)
    
    def __getitem__(self, index):
        return self.contents[index], self.labels[index]

    def __len__(self):
        return len(self.contents)
    
def getDataLoader(seq_len, batch_size):
    word2id = getWord2Id()
    
    train_dataset = TextDataset(train_data, word2id, seq_len=seq_len)
    validation_dataset = TextDataset(validation_data, word2id, seq_len=seq_len)
    test_dataset = TextDataset(test_data, word2id, seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, validation_loader, test_loader
    
def cheat(seq_len, batch_size):
    word2id = getWord2Id()
    
    train_dataset = TextDataset(train_data, word2id, seq_len=seq_len)
    validation_dataset = TextDataset(validation_data, word2id, seq_len=seq_len)
    test_dataset = TextDataset(test_data, word2id, seq_len=seq_len)
    
    concat_dataset = ConcatDataset([train_dataset, validation_dataset, test_dataset])
    
    cheat_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True)
    
    return cheat_loader