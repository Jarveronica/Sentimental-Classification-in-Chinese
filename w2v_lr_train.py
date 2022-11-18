import os
import numpy as np
import pandas as pd
import jieba
import torch
from torchtext.vocab import GloVe, Vectors
import torchtext
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
import joblib

os.chdir('/home/chenyilong/yilongchen/weibo_final')
path = './data'
output_path = './w2v_lr_weibo_output/'
if not os.path.exists(output_path):
    os.mkdir(output_path)

device = 'cpu'


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


# 模型，输入句子rev+标签lab，输出编码后的句子rev+标签lab
class Model(torch.nn.Module):
    def __init__(self,vocab=TEXT.vocab):
        super().__init__()
        self.maxlen = 128
        self.embed_dim = 100
        self.vocab = vocab

        self.embed = torch.nn.Embedding(num_embeddings=len(vocab), embedding_dim=self.embed_dim)
        self.embed.weight.data.copy_(vocab.vectors)

    def forward(self, rev, lab):

        with torch.no_grad():
            rev = self.embed(rev).flatten(1)

        return rev, lab


def train(Model,modelname):

    epoch_n = 1
    batch_n = 500

    # 下游任务模型实例化
    model0 = Model()
    model0.to(device)
    model = SGDClassifier(loss='log', warm_start=True, l1_ratio=0.4)  # loss='hinge'即为linearSVC
    loss_func = torch.nn.CrossEntropyLoss()
    print('model {} loaded!'.format(modelname))

    batch_size = 16
    loader = torch.utils.data.DataLoader(dataset=Dataset('train'), batch_size=batch_size, shuffle=True,
                                         drop_last=True)

    for epoch in range(epoch_n):


        print('-----------epoch:%d-----------' % (epoch+1,))
        accu_sum = 0

        for batch, (rev, lab) in enumerate(loader):

            batch += 1

            rev, lab = model0(rev,lab)

            if (epoch==0)and(batch==1): model = model.fit(rev, lab)
            model = model.partial_fit(rev, lab)
            pred = model.predict(rev)
            accu = (pred == lab.numpy()).sum() / 16
            loss = loss_func(torch.tensor(model.predict_proba(rev)),lab)

            # 计算累计accuracy
            accu_sum = (accu_sum*(batch-1) + accu)/batch


            if batch % 10 == 0:
                print('batch:%d, loss:%.8f, accu:%.8f' % (batch, loss,accu_sum))

                if batch % (batch_n*10) == 0:  # 只训练batch_n*10个batch
                    break
        test(model0,model,modelname)

    joblib.dump(model, output_path + 'weibo_lr_{}_model.pkl'.format(modelname))


def test(model0,model,modelname):

    batch_size = 16

    loader = torch.utils.data.DataLoader(dataset=Dataset('test'), batch_size=batch_size, shuffle=True,
                                         drop_last=True)
    y_pred = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)

    for batch, (rev, lab) in enumerate(loader):

        rev, lab = model0(rev, lab)

        y_pred = np.hstack([y_pred,model.predict(rev)])
        y_true = np.hstack([y_true,lab.numpy()])

    report = classification_report(y_true, y_pred, digits=4)
    result = str(report)
    print(result)
    with open(output_path+'train_w2v_lr_{}_result.txt'.format(modelname),'w',encoding='utf-8') as f:
        f.write(result)


if __name__ == '__main__':
    train(Model,'0')