import tensorflow as tf
import sys
import time
import random
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
import pickle as pk
tf.reset_default_graph()

def seq2id(seq, vocab):
    sentence_id = []
    for word in seq:
        if word not in vocab:
            word = '<UNK>'
        sentence_id.append(vocab[word])
    return sentence_id


class BiLSTM_ADD_CRF():
    def __init__(self,parameter):
        self.batch_size = parameter['batch_size']
        self.epoch_num = parameter['epoch']
        self.hidden_dim = parameter['hidden_dim']
        self.embeddings = parameter['embeddings']
        self.update_embedding = parameter['update_embedding']
        self.dropout_keep_prob = parameter['dropout']
        self.optimizer = parameter['optimizer']
        self.lr = parameter['lr']
        self.clip_grad = parameter['clip']
        self.tag2label = parameter['tag2label']
        self.num_tags = parameter['num_tags']
        self.vocab = parameter['vocab']
        self.shuffle = parameter['shuffle']
        self.model_path = parameter['model_path']
        self.config = tf.ConfigProto()
        
    def batch_yield(self,data):
        # 构建一个迭代器，获取相应的seqs（index型）和label，按照batch_size提取
        if self.shuffle:
            random.shuffle(data)
        seqs,labels = [],[]
        for (seq,label) in data:
            seq = seq2id(seq,self.vocab)
            label = [self.tag2label[label_] for label_ in label]
            if len(seqs) == self.batch_size:
                seq_len_list = [len(i) for i in seqs]
                max_len = max(seq_len_list)
                # tensorflow 1 对于不定长处理不好，placeholder输入需要等长，故进行补0
                seqs = [i+[0]*(max_len-len(i)) for i in seqs]
                labels = [i+[0]*(max_len-len(i)) for i in labels]
                yield seqs,labels,seq_len_list
                seqs,labels = [],[]
            seqs.append(seq)
            labels.append(label)
        if len(seqs) != 0:
            seq_len_list = [len(i) for i in seqs]
            max_len = max(seq_len_list)
            # tensorflow 1 对于不定长处理不好，placeholder输入需要等长，故进行补0
            seqs = [i+[0]*(max_len-len(i)) for i in seqs]
            labels = [i+[0]*(max_len-len(i)) for i in labels]
            yield seqs, labels, seq_len_list
            
    def pre_placeholders(self):
        # 提前设置预留的seqs和labels，和各batch的序列长度，以便后续喂参数
        self.seqs = tf.placeholder(tf.int32, shape=[None, None], name="seqs")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        
    def lookup_layer(self):
        # 从self.embeddings,提取对应index的embedding
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            # 寻找_word_embeddings矩阵中分别为seqs中元素作为下标的值，提取并组合成一个新的向量
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.seqs,
                                                     name="word_embeddings")
        # 模型实现过程设置test3、test4进行过程中变量的输出，查看结构和预期是否一致
        self.test3 = word_embeddings
        self.test4 = self.seqs
        # dropout函数是为了防止在训练中过拟合的操作
        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout_pl)
        
        
    def biLSTM_layer(self):
        # 双向lstm就是两个lstm组成
        with tf.variable_scope("bi-lstm"):
            # 准备前向cell
            cell_fw = LSTMCell(self.hidden_dim)
            # 准备反向cell
            cell_bw = LSTMCell(self.hidden_dim)
            # 此处的输出，包含一个前向输出的结果和后项输出的结果
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            # 简单查看下前向和后项输出结果的形状
            self.test1 = output_fw_seq
            self.test2 = output_bw_seq
            # 输出结果为隐层的输出，将两者拼接在一起作为最终，双向lstm的输出
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("predict"):
            # 初始化相应的权重和偏置
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])
            # 通过矩阵乘法操作，将结果投影到num_tags维的空间上
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])
            
    def crf_pred(self):
        # 大家可以查下tensorflow里面关于crf_log_likelihood这个api
        # crf_log_likelihood 在这里是一个损失函数
        # 输入：
        # inputs，就是每个标签的预测概率值，就是经过矩阵乘法变换得到的self.logits
        # tag_indices，是期望输出，target
        # sequence_lengths，和上面一样，是样本真实的序列长度
        # transition_params，就是李老师讲的转移概率，可以没有，这个函数可以自己计算
        # 输出：
        # log_likelihood，是一个标量，在这里有点类似于-损失
        # transition_params，函数自己计算的转移概率，在我们推理阶段就是用这个
        log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                   tag_indices=self.labels,
                                                                   sequence_lengths=self.sequence_lengths)
        self.loss = -tf.reduce_mean(log_likelihood)

        
    def trainstep(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            # 此处按照所需选择不同的优化器，如所提供的论文中所用Adam、RMSprop等
            if self.optimizer in ['Adam','Adadelta','Adagrad','RMSProp']:
                optim = eval('tf.train.'+self.optimizer+'Optimizer(learning_rate=self.lr_pl)')
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            grads_and_vars = optim.compute_gradients(self.loss)
            # 此处和之前在RNN、LSTM处一致，对梯度进行修饰
            # tf.clip_by_value(A, min, max)指将列表A中元素压缩在min和max之间，大于max或小于min的值改成max和min
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            # 梯度更新
            self.train = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)
        
    def init(self):
        self.init = tf.global_variables_initializer()
        
    def have_a_test(self,data):
        self.pre_placeholders()
        self.lookup_layer()
#         self.biLSTM_layer()
#         self.crf_pred()
#         self.trainstep()
        self.init()
        #test data
        data = self.batch_yield(data)
        data_seqs,data_labels,seq_len_list = next(data)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config=self.config) as sess:
            sess.run(self.init)
            feed_dict = {self.seqs: data_seqs,
                         self.sequence_lengths: seq_len_list,
                         self.labels: data_labels,
                         self.dropout_pl:self.dropout_keep_prob,
                         self.lr_pl:self.lr
                        }
            return sess.run([self.test3,self.test4],feed_dict=feed_dict),seq_len_list
        
    def train_one_epoch(self,sess,epoch,data):
        # 训练批次数
        num_batches = (len(data) + self.batch_size - 1) // self.batch_size
        # 记录时间
        sta_time = time.time()
        # 数据迭代器
        batches = self.batch_yield(data)
        cal_loss = 0
        for step,(seqs,labels,seq_len_list) in enumerate(batches):
            feed_dict = {self.seqs: seqs,
                         self.sequence_lengths: seq_len_list,
                         self.labels: labels,
                         self.dropout_pl:self.dropout_keep_prob,
                         self.lr_pl:self.lr
                        }
            _, loss_train,  step_num_ = sess.run([self.train, self.loss, self.global_step],feed_dict=feed_dict)
            cal_loss += loss_train
            sys.stdout.write(' processing: {} epoch / {} batch / {} batches / {} loss / {} time'.format(epoch, step + 1, num_batches ,cal_loss/(step+1) ,time.time()-sta_time) + '\r')
        print('\n')
        
    def build_graph(self):
        self.pre_placeholders()
        self.lookup_layer()
        self.biLSTM_layer()
        self.crf_pred()
        self.trainstep()
        self.init()
        
    def train(self,train_data):
        self.build_graph()
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config=self.config) as sess:
            sess.run(self.init)
            # 循环训练epoch_num次
            for epoch in range(self.epoch_num):
                self.train_one_epoch(sess, epoch, train_data)
                saver.save(sess, self.model_path, global_step=epoch)
                
    #预测一批数据集
    def predict_one_batch(self, sess, seqs, seq_len_list):
        #将样本进行整理（填充0方式使得每句话长度一样，并返回每句话实际长度）
        feed_dict = {self.seqs: seqs,
                         self.sequence_lengths: seq_len_list,
                         self.dropout_pl:1.0,
                        }
        logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
        label_list = []
        for logit, seq_len in zip(logits, seq_len_list):
            # 维特比算法（解码），通过转移概率递推，计算最优路径，具体可参考李老师的ppt或后续有关于crf的实现
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
            label_list.append(viterbi_seq)
        return label_list

def load_model():
    parameter = pk.load(open('model/ner/parameter.pkl','rb'))
    tf.reset_default_graph()
    tag2label = parameter['tag2label']
    model_path = parameter['model_path']
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print('ckpt_file=',ckpt_file)
    #加载图
    model = BiLSTM_ADD_CRF(parameter)
    model.build_graph()
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto())
    saver.restore(sess, ckpt_file)
    label2tag = dict(zip(tag2label.values(),tag2label.keys()))
    return sess,model,label2tag

def for_pred(ins,model,sess,label2tag):
    for_pred = list(ins.strip())#例：['在', '弄', '恩', '哦', '呜']
    for_pred = [(for_pred, ['O'] * len(for_pred))]
    for_pred_batch,_,for_pred_len = next(model.batch_yield(for_pred))
    res = model.predict_one_batch(sess,for_pred_batch,for_pred_len)[0]
    entity_list = []
    pad = -1
    tmp = None
    for ind,i in enumerate(res):
        if ind <= pad:
            continue
        i = label2tag[i] 
        if i[0] in ['S','B']:
            if i[0] == 'S':
                entity_list.append([for_pred[0][0][ind],i.split('-')[1],[ind]])
            else:
                tmp = [for_pred[0][0][ind],i.split('-')[1],[ind]]
                for j in range(ind+1,len(res)):
                    j_tag = label2tag[res[j]]
                    if j_tag == 'O':
                        pad = ind
                        tmp = None
                        break
                    if j_tag.split('-')[1] == tmp[1] and j_tag[0] != 'B':
                        tmp[0] += for_pred[0][0][j]
                        tmp[-1].append(j)
                        if j_tag[0] == 'E':
                            pad = j
                            entity_list.append(tmp)
                            tmp = None
                            break
                        if j_tag[0] == 'I':
                            pad = j
                    else:
                        pad = ind
                        tmp = None
                        break
    return entity_list,res,for_pred[0][0]


ner_sess,ner_model,ner_label2tag = load_model()