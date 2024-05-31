import torch
import torch.nn as nn
import torch.nn.functional as F
from data import *

word2id = getWord2Id() # 训练集词库

class ModelConfig:
    # 模型参数
    embed_size = 50 # 词嵌入维度
    hidden_size = 128 # 隐藏层维度
    hidden_layers_num = 2 # 隐藏层数量
    dropout_rate = 0.5 # dropout比例
    kernel_num = 32 # 卷积核数量
    kernel_sizes = [3, 5, 7] # 卷积核高度，宽度恒为词向量大小
    vocabulary_size = len(word2id) + 1 # 词汇表大小，从1开始
    seq_len = 50 # 句子最大长度
    bidirectional = True # 是否双向LSTM
    num_directions = 2 # LSTM方向数量
    
class TextCNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super(TextCNN, self).__init__()
        self.config = config
        
        self.__name__ = "TextCNN"
        
        self.embedding = nn.Embedding(config.vocabulary_size, config.embed_size) # 初始词嵌入
        self.embedding.weight.requires_grad = True # 词嵌入也需要训练
        self.embedding.weight.data.copy_(torch.from_numpy(getWord2Vec(word2id))) # 初始化词嵌入
        
        self.conv1 = nn.Conv2d(1, config.kernel_num, (config.kernel_sizes[0], config.embed_size)) # 第一个卷积层
        self.conv2 = nn.Conv2d(1, config.kernel_num, (config.kernel_sizes[1], config.embed_size)) # 第二个卷积层
        self.conv3 = nn.Conv2d(1, config.kernel_num, (config.kernel_sizes[2], config.embed_size)) # 第三个卷积层
        
        self.dropout = nn.Dropout(config.dropout_rate) # 随机丢弃
        
        self.fc = nn.Linear(config.kernel_num * len(config.kernel_sizes), 2) # 全连接层
        
    def forward(self, x):
        # x: [batch_size, sentence_len]
        x = self.embedding(x).unsqueeze(1) # 词嵌入，并扩展为一个通道
        # x: [batch_size, 1, sentence_len, embed_size]
        # 卷积核的宽度为词向量的宽度，因此卷积后shape为
        # [batch_size, kernel_num, sentence_len - kernel_size + 1, 1]
        # 将最后一个维度去除之后在进行池化操作
        # 同理池化后也要去除一维操作
        x1 = F.relu(self.conv1(x).squeeze(3)) # 第一个卷积层
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2) # 最大池化
        x2 = F.relu(self.conv2(x).squeeze(3)) # 第二个卷积层
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2) # 最大池化
        x3 = F.relu(self.conv3(x).squeeze(3)) # 第三个卷积层
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2) # 最大池化
        # 将三个filter得到的结构拼接起来最为最终结果，随机失活后进入全连接层，经过softmax取对数后输出
        y = self.fc(self.dropout(torch.cat((x1, x2, x3), dim=1)))
        return F.log_softmax(y, dim=1) # 分类
    
class LSTM(nn.Module):
    def __init__(self, config: ModelConfig):
        super(LSTM, self).__init__()
        self.config = config
        
        self.__name__ = "LSTM"
        
        self.embedding = nn.Embedding(config.vocabulary_size, config.embed_size) # 初始词嵌入
        self.embedding.weight.requires_grad = True # 词嵌入也需要训练
        self.embedding.weight.data.copy_(torch.from_numpy(getWord2Vec(word2id))) # 初始化词嵌入
        
        # 输入[batch_size, seq_len, embed_size]
        # 输出[batch_size, seq_len, hidden_size * num_directions]
        self.encoder = nn.LSTM(
            input_size = config.embed_size,
            hidden_size = config.hidden_size,
            num_layers = config.hidden_layers_num,
            bidirectional = config.bidirectional, # 双向LSTM
            batch_first = True # 改变输入格式维度为[batch_size, seq_len, embed_size]
        )
        self.decoder = nn.Linear(config.seq_len * config.hidden_size * config.num_directions, 2) # 输出层
        
    def forward(self, x):
        # x: [batch_size, sentence_len]
        x = self.embedding(x) # 词嵌入
        # x: [batch_size, sentence_len, embed_size]
        output, (h_n, s_n) = self.encoder(x)
        # output: [batch_size, sentence_len, hidden_size * num_directions]
        output = output.reshape(output.size(0), -1) # 展平
        y = self.decoder(output)
        return F.log_softmax(y, dim=1)
    
class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super(MLP, self).__init__()
        self.config = config
        
        self.__name__ = "MLP"
        
        print(config.vocabulary_size)
        print(config.embed_size)
        
        self.embedding = nn.Embedding(config.vocabulary_size, config.embed_size) # 初始词嵌入
        self.embedding.weight.requires_grad = True # 词嵌入也需要训练
        self.embedding.weight.data.copy_(torch.from_numpy(getWord2Vec(word2id))) # 初始化词嵌入
        
        self.linear1 = nn.Linear(config.embed_size * config.seq_len, config.hidden_size) # 第一个全连接层
        self.linear2 = nn.Linear(config.hidden_size, (int)(config.hidden_size / 2)) # 第二个全连接层
        self.linear3 = nn.Linear((int)(config.hidden_size / 2), (int)(config.hidden_size / 4)) # 第三个全连接层
        self.linear4 = nn.Linear((int)(config.hidden_size / 4), 2) # 第四个全连接层
        
        self.dropout = nn.Dropout(config.dropout_rate) # 随机丢弃
        
    def forward(self, x):
        print(x)
        x = self.embedding(x).view(x.size(0), -1)# 词嵌入
        x = F.relu(self.linear1(x)) # 第一个全连接层
        x = self.dropout(x)
        x = F.relu(self.linear2(x)) # 第二个全连接层
        x = self.dropout(x)
        x = F.relu(self.linear3(x)) # 第三个全连接层
        x = self.dropout(x)
        y = self.linear4(x) # 第四个全连接层
        return F.log_softmax(y, dim=1)
        