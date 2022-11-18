import logging
import os
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

os.chdir('/home/chenyilong/yilongchen/weibo_example/word2vec/')

def train():
    # 日志信息输出
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info('running %s' % ' '.join(sys.argv))
    dimensions = [100, 200, 300]
    # 训练模型 生成不同维度的词向量
    for dimension in dimensions:
        input_file = './weibo_train_corpus_zh.txt'
        # output1 = './weibo_zh_word2vec_' + str(dimension) + '.model'
        output2 = './weibo_zh_word2vec_format_' + str(dimension) + '.txt'
        '''
        vector_size 词向量的长度
        window 最大距离
        min_count 出现次数少于该数的不考虑
        cbow_mean=1 使用CBOW模型
        negative=5 使用negative sampling
        hs=0 不使用hierarchical softmax
        '''
        model = Word2Vec(LineSentence(input_file), vector_size=dimension, window=5, min_count=2
                         , workers=multiprocessing.cpu_count(), epochs=15)
        # model.save(output1)
        model.wv.save_word2vec_format(output2, binary=False)


if __name__ == '__main__':
    train()
