import os
import numpy as np
import pandas as pd
import jieba
import torch
from sklearn.metrics import classification_report
from transformers import BertModel, BertTokenizer, AdamW

os.chdir('/home/chenyilong/yilongchen/weibo_final')
path = './data'
output_path = './bert_only_weibo_output/'
if not os.path.exists(output_path):
    os.mkdir(output_path)

# 获取bert预训练token和pretrained模型
device = 'cuda:5'

# pretrained_model = 'hfl/chinese-bert-wwm-ext'
pretrained_model = 'bert-base-chinese'
token = BertTokenizer.from_pretrained(pretrained_model)
pretrained = BertModel.from_pretrained(pretrained_model).to(device)


# 获取停用词
def get_stopwords():
    stopwords = []
    with open('./word2vec/hit_stopwords.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords

# 分词器
def get_segment_words(text):
    # 分词处理
    words = []
    stopwords = get_stopwords()
    sentence = jieba.lcut(text, cut_all=False)
    for word in sentence:
        if word != ' ' and word not in stopwords:
            words.append(word)
    sentence = ''.join(words)

    return sentence


# 数据集，输入'train'，'dev'或'test'，输出句子rev+标签lab
class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode):
        self.rev = pd.read_csv(os.path.join(path, './{}.csv'.format(mode)))['review'].apply(get_segment_words).values
        self.lab = pd.read_csv(os.path.join(path, './{}.csv'.format(mode)))['label'].values
        self.lab = torch.tensor(self.lab).to(device)
        print('{} data loaded!'.format(mode))

    def __len__(self):
        return len(self.rev)

    def __getitem__(self, i):
        rev = self.rev[tuple([i])]
        lab = self.lab[tuple([i])]
        return rev,lab

# 模型，输入句子rev+标签lab，输出损失loss+准确率accu+预测pred
class Word_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxlen = 128
        self.insize = 768*self.maxlen
        # self.midsize1 = 1024
        # self.midsize2 = 128
        self.outsize = 2
        # self.fc1 = torch.nn.Linear(self.insize, self.midsize1)
        # self.fc2 = torch.nn.Linear(self.midsize1, self.midsize2)
        # self.fc3 = torch.nn.Linear(self.midsize2, self.outsize)
        self.fc4 = torch.nn.Linear(self.insize, self.outsize)
        self.tanh = torch.nn.ReLU() # 整流单元，非线性变换
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, rev, lab):

        with torch.no_grad():
            token_all = token.batch_encode_plus(batch_text_or_text_pairs=rev, truncation=True,
                                                padding='max_length',max_length=self.maxlen, return_tensors='pt',
                                                return_length=True)

            id = token_all['input_ids']
            mask = token_all["attention_mask"]
            rev = pretrained(input_ids=id.to(device),attention_mask=mask.to(device)).last_hidden_state

        rev = rev.flatten(1)
        rev = self.fc4(rev)
        rev = self.tanh(rev)


        loss = self.loss(rev,lab)

        with torch.no_grad():

            pred = torch.max(rev,1)[1]
            accu = (pred == lab).sum() / 16


        return loss, accu, pred

class Sentence_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxlen = 128
        self.insize = 768
        # self.midsize1 = 1024
        # self.midsize2 = 128
        self.outsize = 2
        # self.fc1 = torch.nn.Linear(self.insize, self.midsize1)
        # self.fc2 = torch.nn.Linear(self.midsize1, self.midsize2)
        # self.fc3 = torch.nn.Linear(self.midsize2, self.outsize)
        self.fc4 = torch.nn.Linear(self.insize, self.outsize)
        self.tanh = torch.nn.ReLU() # 整流单元，非线性变换
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, rev, lab):

        with torch.no_grad():
            token_all = token.batch_encode_plus(batch_text_or_text_pairs=rev, truncation=True,
                                                padding='max_length',max_length=self.maxlen, return_tensors='pt',
                                                return_length=True)

            id = token_all['input_ids']
            mask = token_all["attention_mask"]
            rev = pretrained(input_ids=id.to(device),attention_mask=mask.to(device)).last_hidden_state[:,0]  # 仅取用[CLS]

        # rev = rev.flatten(1)
        rev = self.fc4(rev)
        rev = self.tanh(rev)


        loss = self.loss(rev,lab)

        with torch.no_grad():

            pred = torch.max(rev,1)[1]
            accu = (pred == lab).sum() / 16


        return loss, accu, pred


def train(Model,modelname,lr=5e-4):

    model_path = output_path+'/bert_only_{}.model'.format(modelname)

    epoch_n = 10
    batch_n = 5000

    # 下游任务模型实例化
    model = Model()
    model.to(device)
    model.train()

    # 参数初始化
    def weight_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    # model.apply(weight_init)

    batch_size = 16
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(dataset=Dataset('train'), batch_size=batch_size, shuffle=True,
                                         drop_last=True)

    for epoch in range(epoch_n):


        print('-----------epoch:%d-----------' % (epoch+1,))
        accu_sum = 0

        for batch, (rev, lab) in enumerate(loader):

            batch += 1

            loss = model(rev,lab)[0]
            accu = model(rev,lab)[1]
            loss.backward()

            # 计算累计accuracy
            accu_sum = (accu_sum*(batch-1) + accu)/batch

            optimizer.step()
            optimizer.zero_grad()

            if batch % 10 == 0:
                print('batch:%d, loss:%.8f, accu:%.8f' % (batch, loss,accu_sum))

                if batch % (batch_n*10) == 0:  # 只训练batch_n*10个batch
                    break
        test(model,modelname)

    torch.save(model, model_path)

def test(model,modelname):

    batch_size = 16

    loader = torch.utils.data.DataLoader(dataset=Dataset('test'), batch_size=batch_size, shuffle=True,
                                         drop_last=True)
    y_pred = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)

    for batch, (rev, lab) in enumerate(loader):

        y_pred = torch.concat([y_pred,model(rev,lab)[2]])
        y_true = torch.concat([y_true,lab])

    report = classification_report(y_true.to('cpu'), y_pred.to('cpu'), digits=4)
    result = str(report)
    print(result)
    with open(output_path+'train_bert_only_{}_result.txt'.format(modelname),'w',encoding='utf-8') as f:
        f.write(result)


if __name__ == '__main__':
    train(Sentence_Model,'sentence')

