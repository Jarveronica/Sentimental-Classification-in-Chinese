# 中文情感分类 Sentimental-Classification-in-Chinese

## 项目介绍：一个使用Word2Vec和BERT/CNN/LSTM模型进行中文情感分类的初级NLP任务。
## Introduction: A beginner-level NLP task using Word2Vec and BERT/CNN/LSTM models for sentiment classification.
基于[weibo_senti_100k]数据集，分别使用Word2Vec、Glove和BERT_pretrained的方式构造词向量，并使用逻辑回归、LSTM、CNN、BERT等模型进行情感分类任务的训练。
Based on the [weibo_senti_100k] dataset, word vectors were constructed using Word2Vec, Glove, and BERT_pretrained, respectively, and logistic regression, LSTM, CNN, and BERT models were used to train the sentiment classification task.

## 环境配置 Configuration
cudatoolkit               11.3.1
gensim                    4.2.0
jieba                     0.42.1
numpy                     1.23.4
pandas                    1.4.4
python                    3.9.12
pytorch                   1.12.1
scikit-learn              1.0.2
torch                     1.12.1+cu113
torchtext                 0.6.0
transformers              4.24.0
