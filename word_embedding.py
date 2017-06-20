#encoding:utf-8

from collections import Counter

import numpy as np

from gensim.models.word2vec import Word2Vec




def build_sentence_vector(text, size,imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

#计算词向量
def get_train_vecs(x_train):
    n_dim = 300
    #初始化模型和词表
    imdb_w2v = Word2Vec(size=n_dim,window=5,min_count=10)
    imdb_w2v.build_vocab(x_train)
    
    #在评论训练集上建模(可能会花费几分钟)
    imdb_w2v.train(total_examples=imdb_w2v.corpus_count,epochs=imdb_w2v.iter,sentences=x_train)
    
    train_vecs = np.concatenate([build_sentence_vector(z, n_dim,imdb_w2v) for z in x_train])
    #train_vecs = scale(train_vecs)
    
    np.save('user_vecs.npy',train_vecs)
    print train_vecs.shape
    return imdb_w2v,train_vecs


#输入一个句子（子的列表），返回每个字的词频（计数）
def counter_words(list1):
    countDict=dict(Counter(list1))
    return countDict
