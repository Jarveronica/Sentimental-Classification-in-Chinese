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

## 数据准备 Data Preparation
数据集存放在data文件夹下，名为weibo\_senti\_100k.csv，运行代码train\_valid\_test_split.py生成train.csv、dev.csv、test.csv文件。训练集、验证集、测试集的拆分比例为8:1:1。
Word2Vec Embedding存放在Word2Vec文件夹下，运行word2vec_train.py即可。

The dataset is stored in the data folder named weibo\_senti\_100k.csv, and the code train\_valid\_test_split.py is run to generate the train.csv, dev.csv, and test.csv files. The splitting ratio of training set, validation set, and test set is 8:1:1.
Word2Vec Embedding is stored in the Word2Vec folder, just run word2vec_train.py.

## 参考 Reference
数据集划分和Word2Vec部分代码来自于：https://github.com/wyd-case/weibo_sentiment_analysis-master 。
The dataset partitioning and Word2Vec part code is taken from https://github.com/wyd-case/weibo_sentiment_analysis-master.

## 性能测试结果 Performance Test Results
最后，我想在这里给出各种基准的性能测试结果。
Lastly, I would like to give the performance test results of various benchmarks here.

<img width="400" alt="image" src="https://user-images.githubusercontent.com/98510598/206902081-26329ec7-dfc4-4877-a54f-6012034f1fcf.png">
