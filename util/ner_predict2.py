from collections import defaultdict
from operator import itemgetter
import numpy as np
import torch
import torch.nn.functional as F # pytorch 激活函数的类
import pickle as pk
import pandas as pd
from torch import nn
from tqdm import tqdm
from torchcrf import CRF


# 构建基于bilstm+crf实现ner
class bilstm_crf(nn.Module):
    def __init__(self, parameter):
        super(bilstm_crf, self).__init__()
        vocab_size = parameter['vocab_size']
        embedding_dim = parameter['d_model']
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        hidden_size = parameter['hid_dim']
        num_layers = parameter['n_layers']
        dropout = parameter['dropout']
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)

        output_size = parameter['num_tags']
        self.fc = nn.Linear(hidden_size*2, output_size)
        
        self.crf = CRF(output_size,batch_first=True)
        
    def forward(self, x):
        out = self.embedding(x)
        out,(h, c)= self.lstm(out)
        out = self.fc(out)
        return out

# 此处是加载对应的模型和配置文件
# def load_model(mode_path):
#     parameter = pk.load(open(mode_path+'parameter.pkl','rb'))
#     #     parameter['device'] = torch.device('cpu')
#     # 因为bert模型需要加载他对应的config文件，因此此处进行了一定的区分
#     model = bilstm_crf(parameter).to(parameter['device'])
#     model.load_state_dict(torch.load(model_path+'bilstm_crf.h5'))
#     model.eval() 
#     return model,parameter
# 此处是加载对应的模型和配置文件
def load_model(mode_path):
    parameter = pk.load(open(mode_path+'parameter.pkl','rb'))
    parameter['device'] = torch.device('cpu')
    # 因为bert模型需要加载他对应的config文件，因此此处进行了一定的区分
    model = bilstm_crf(parameter).to(parameter['device'])
    model.load_state_dict(torch.load(model_path+'bilstm_crf.h5',map_location='cpu'))
    model.eval() 
    return model,parameter

def keyword_predict(input):
    def list2torch(ins):
        return torch.from_numpy(np.array(ins))
    def seq2id(seq, vocab):
        sentence_id = []
        for word in seq:
            if word not in vocab:
                word = '<UNK>'
            sentence_id.append(vocab[word])
        return sentence_id
    input = list(input)
    ind2key = dict(zip(parameter['tag2label'].values(),parameter['tag2label'].keys()))
    input_id = seq2id(input,parameter['vocab'])#itemgetter(*input)(parameter['word2ind'])
    print(input_id)
    predict = model.crf.decode(model(list2torch([input_id]).long().to(parameter['device'])))[0]
    predict = itemgetter(*predict)(ind2key)
    print(predict)
    keys_list = []
    for ind,i in enumerate(predict):
        if i == 'O':
            continue
        if i[0] == 'S':
            if not(len(keys_list) == 0 or keys_list[-1][-1]):
                del keys_list[-1]
            keys_list.append([input[ind],[i],[ind],True])
            continue
        if i[0] == 'B':
            if not(len(keys_list) == 0 or keys_list[-1][-1]):
                del keys_list[-1]
            keys_list.append([input[ind],[i],[ind],False])
            continue
        if i[0] == 'I':
            if len(keys_list) > 0 and not keys_list[-1][-1] and \
            keys_list[-1][1][0].split('-')[1] == i.split('-')[1]:
                keys_list[-1][0] += input[ind]
                keys_list[-1][1] += [i]
                keys_list[-1][2] += [ind]
            else:
                if len(keys_list) > 0:
                    del keys_list[-1]
            continue
        if i[0] == 'E':
            if len(keys_list) > 0 and not keys_list[-1][-1] and \
            keys_list[-1][1][0].split('-')[1] == i.split('-')[1]:
                keys_list[-1][0] += input[ind]
                keys_list[-1][1] += [i]
                keys_list[-1][2] += [ind]
                keys_list[-1][3] = True
            else:
                if len(keys_list) > 0:
                    del keys_list[-1]
            continue
    keys_list = [[i[0],i[1][0].split('-')[1],i[2]] for i in keys_list]
    return keys_list

model_path = 'model/ner/'
model,parameter = load_model(model_path)
# keyword_predict('李白是谁')