import jieba as jie
import pandas as pd
import numpy as np
import pickle
from gensim.models import Word2Vec
import re
from tqdm import tqdm


def get_corpus(file_path, header_idx=0):
    src_df = pd.read_excel(file_path, header=header_idx)
    print('Corpus shape before: ', src_df.shape)
    src_df = src_df.dropna(subset=['Response'])
    print('Corpus shape after: ', src_df.shape)
    return src_df['Question'].tolist(), src_df['Response'].tolist()


def clean_text(text):
    text = re.sub(
        u"([hH]ttp[s]{0,1})://[a-zA-Z0-9\.\-]+\.([a-zA-Z]{2,4})(:\d+)?(/[a-zA-Z0-9\-~!@#$%^&*+?:_/=<>.',;]*)?", '',
        text)  # remove http:xxx
    text = re.sub(u'#[^#]+#', '', text)  # remove #xxx#
    text = re.sub(u'回复@[\u4e00-\u9fa5a-zA-Z0-9_-]{1,30}:', '', text)  # remove "回复@xxx:"
    text = re.sub(u'@[\u4e00-\u9fa5a-zA-Z0-9_-]{1,30}', '', text)  # remove "@xxx"
    text = re.sub(r'[0-9]+', 'DIG', text.strip()).lower()
    text = ''.join(text.split())  # split remove spaces
    return text


def word_tokenize(line):
    content = clean_text(line)
    #content_words = [m for m in jie.lcut(content) if m not in self.stop_words]
    return jie.lcut(content)


def load_embedding(cached_embedding_file):
    """load embeddings"""
    with open(cached_embedding_file, mode='rb') as f:
        return pickle.load(f)


def save_embedding(word_embeddings, cached_embedding_file):
    """save word embeddings"""
    with open(cached_embedding_file, mode='wb') as f:
        pickle.dump(word_embeddings, f)


def get_word_embedding_matrix(word2idx, pretrained_embeddings_file, embedding_dim=200):
    """Load pre-trained embeddings"""
    # initialize an empty array
    pre_trained_embeddings = np.zeros((len(word2idx), embedding_dim))
    initialized = 0
    exception = 0
    num = 0
    with open(pretrained_embeddings_file, mode='r',encoding='utf-8') as f:
        try:
            for line in f:
                word_vec = line.split()
                idx = word2idx.get(word_vec[0], -1)
                # if current word exists in word2idx
                if idx != -1:
                    pre_trained_embeddings[idx] = np.array(word_vec[1:], dtype=np.float)
                    initialized += 1

                num += 1

                if num % 10000 == 0:
                    print(num)
        except:
            exception += 1

    print('Pre-trained embedding initialization proportion: ', (initialized + 0.0) / len(word2idx))
    print('exception num: ', exception)
    return pre_trained_embeddings


def build_word_embedding(corpus, gensim_model_path, gensim_word_embdding_path):

    # build the model
    model = Word2Vec(min_count=1, size=100, window=5, sg=1, negative=5, sample=0.001,
                     iter=30)  # initialize an empty model
    model.build_vocab(sentences=corpus)
    model.train(sentences=corpus, total_examples=model.corpus_count, epochs=model.iter)

    # save the word vectors
    model.wv.save_word2vec_format(gensim_word_embdding_path, binary=False)
    # save the model
    model.save(gensim_model_path)

    print('\nGensim model build successfully!')

    print('\nTest the performance of word2vec model')

    for test_word in ['门禁卡', '食堂', '试用期']:
        aa = model.most_similar(test_word)[0:10]
        print('\nMost similar word of %s is:' % test_word)
        for word, score in aa:
            print('{} {}'.format(word, score))


    '''
    # save word counts
    sorted_word_counts = OrderedDict(sorted(model.wv.vocab.items(), key=lambda x: x[1].count, reverse=True))
    word_counts_file = codecs.open('./word_counts.txt', mode='w', encoding='utf-8')
    for k, v in sorted_word_counts.items():
        word_counts_file.write(k + ' ' + str(v.count) + '\n')
    '''
