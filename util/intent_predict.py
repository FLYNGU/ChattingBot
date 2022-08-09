import torch
import numpy as np
import pandas as pd
import pickle as pk
from tqdm import tqdm
import torch.nn.functional as F # pytorch 激活函数的类
from torch import nn,optim # 构建模型和优化器
from operator import itemgetter
from collections import defaultdict

# 构建分类模型
class TextRNN(nn.Module):
    def __init__(self, parameter):
        super(TextRNN, self).__init__()
        embedding_dim = parameter['embedding_dim']
        hidden_size = parameter['hidden_size']
        output_size = parameter['output_size']
        num_layers = parameter['num_layers']
        dropout = parameter['dropout']
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size*2, output_size)
        
    def forward(self, x):
        out,(h, c)= self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
def load_model(path):
    parameter = pk.load(open(path,'rb'))
    parameter['dropout'] = 0
    model = TextRNN(parameter).to(parameter['cuda'])
    model.load_state_dict(torch.load(parameter['model_path']+'model-rnn.h5'))
    return parameter,model

def batch_predict(chars,parameter):
        max_len = 0
        batch_x = []
        for iters in range(len(chars)):
            for i in range(len(chars[iters])):
                if chars[iters][i] not in parameter['char2ind']:
                    chars[iters][i] = '<unk>'
            batch_ids = itemgetter(*chars[iters])(parameter['char2ind'])
            try:
                batch_ids = list(batch_ids)
            except:
                batch_ids = [batch_ids,0]
            if len(batch_ids) > max_len:
                max_len = len(batch_ids)
            batch_x.append(batch_ids)
        batch_x = [np.array(list(itemgetter(*x_ids)(parameter['ind2embeding']))+[parameter['ind2embeding'][0]]*(max_len-len(x_ids))) for x_ids in batch_x]
        device = parameter['cuda']
        return torch.from_numpy(np.array(batch_x)).to(device)
    
def predict(ins,model,parameter):
    seqs = batch_predict(ins,parameter)
    res = model(seqs)
    predicted_prob,predicted_index = torch.max(F.softmax(res, 1), 1)
    res = predicted_index.cpu().numpy()
    return res


intent0_parameter,intent0_model = load_model('model/intent0/parameter.pkl')
intent1_parameter,intent1_model = load_model('model/intent1/parameter.pkl')
