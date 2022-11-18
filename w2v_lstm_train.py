import os
import numpy as np
import pandas as pd
import jieba
import torch
from torchtext.vocab import GloVe, Vectors
import torchtext
from sklearn.metrics import classification_report
from transformers import BertModel, BertTokenizer, AdamW

os.chdir('/home/chenyilong/yilongchen/weibo_final')
path = './data'
output_path = './w2v_lstm_weibo_output/'
if not os.path.exists(output_path):
    os.mkdir(output_path)

device = 'cuda:4'



# 获取停用词
def get_stopwords():

    stopwords = []
    with open('./word2vec/hit_stopwords.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    print('stopwords got!')
    return stopwords

# 分词器
def get_segment_words(text,stopwords=get_stopwords()):
    # 分词处理
    words = []
    sentence = jieba.lcut(text, cut_all=False)
    for word in sentence:
        if word != ' ' and word not in stopwords:
            words.append(word)

    return words


# 获取字典
text = pd.read_csv(os.path.join(path, './weibo_senti_100k.csv'))['review'].apply(get_segment_words).tolist()
text = [i for words in text for i in words]
TEXT = torchtext.data.Field(sequential=False)
vectors = Vectors(name='./word2vec/weibo_zh_word2vec_format_100.txt')
# TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=100))
TEXT.build_vocab(text, vectors=vectors)


# 数据集，输入'train'，'dev'或'test'，输出句子rev+标签lab
class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, vocab=TEXT.vocab, max_len=128):
        self.rev = pd.read_csv(os.path.join(path, './{}.csv'.format(mode)))['review'].apply(get_segment_words).tolist()
        self.rev = [[vocab.stoi[i] for i in words[:max_len]] for words in self.rev]
        self.rev = torch.vstack([torch.nn.functional.pad(torch.tensor(words,dtype=int), (0,max_len-len(words))) for words in self.rev]).to(device)
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
class Model(torch.nn.Module):
    def __init__(self,vocab=TEXT.vocab):
        super().__init__()
        self.maxlen = 128
        self.embed_dim = 100
        self.hidsize = 64
        self.insize = self.maxlen*self.hidsize*2  # 双向
        self.outsize = 2
        self.layernum = 2
        self.vocab = vocab

        self.embed = torch.nn.Embedding(num_embeddings=len(vocab), embedding_dim=self.embed_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.fc = torch.nn.Linear(self.insize, self.outsize)
        self.lstm = torch.nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidsize,
                            num_layers=self.layernum, dropout=0.2, batch_first=True, bidirectional=True)
        self.tanh = torch.nn.ReLU() # 整流单元，非线性变换
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, rev, lab):

        with torch.no_grad():
            rev = self.embed(rev)

        rev = self.lstm(rev)[0].flatten(1)
        rev = self.fc(rev)
        rev = self.tanh(rev)


        loss = self.loss(rev,lab)

        with torch.no_grad():

            pred = torch.max(rev,1)[1]
            accu = (pred == lab).sum() / 16


        return loss, accu, pred


def train(Model,modelname,lr=5e-4):

    model_path = output_path+'/w2v_lstm_{}.model'.format(modelname)

    epoch_n = 10
    batch_n = 5000

    # 下游任务模型实例化
    model = Model()
    model.to(device)
    model.train()
    print('model {} loaded!'.format(modelname))

    # 参数初始化
    def weight_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    model.apply(weight_init)

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
    with open(output_path+'train_w2v_lstm_{}_result.txt'.format(modelname),'w',encoding='utf-8') as f:
        f.write(result)


if __name__ == '__main__':
    train(Model,'0')