
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import MatrixSimilarity

from sklearn.feature_extraction.text import TfidfVectorizer
from annoy import AnnoyIndex
import numpy as np


class TFIDFRetrievalModel:
    def __init__(self, corpus):

        '''
        min_freq = 3
        self.dictionary = Dictionary(corpus)
        # Filter low frequency words from dictionary.
        low_freq_ids = [id_ for id_, freq in
                        self.dictionary.dfs.items() if freq <= min_freq]
        self.dictionary.filter_tokens(low_freq_ids)
        self.dictionary.compactify()
        self.corpus = [self.dictionary.doc2bow(line) for line in corpus]
        self.model = TfidfModel(self.corpus)
        self.corpus_mm = self.model[self.corpus]
        self.index = MatrixSimilarity(self.corpus_mm)
        '''

        corpus_str = []
        for line in corpus:
            corpus_str.append(' '.join(line))

        self.tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        sentence_tfidfs = np.asarray(self.tfidf.fit_transform(corpus_str).todense().astype('float'))
        # build search model
        self.t = AnnoyIndex(sentence_tfidfs.shape[1])
        for i in range(sentence_tfidfs.shape[0]):
            self.t.add_item(i, sentence_tfidfs[i, :])
        self.t.build(10)

    def get_top_similarities(self, query, topk=10):
        """query: [word1, word2, ..., wordn]"""
        '''
        query_vec = self.model[self.dictionary.doc2bow(query)]
        scores = self.index[query_vec]
        rtn = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:topk]
        return rtn
        '''
        '''
        query2tfidf = []
        for word in query:
            if word in self.word2tfidf:
                query2tfidf.append(self.word2tfidf[word])
                query2tfidf = np.array(query2tfidf)
        '''
        query2tfidf = np.asarray(self.tfidf.transform([' '.join(query)]).todense().astype('float'))[0]

        top_ids, top_distances = self.t.get_nns_by_vector(query2tfidf, n=topk, include_distances=True)

        return top_ids