from gensim.summarization import bm25


class BM25RetrievalModel:
    """BM25 definition: https://en.wikipedia.org/wiki/Okapi_BM25"""
    def __init__(self, corpus):
        self.model = bm25.BM25(corpus)

    def get_top_similarities(self, query, topk=10):
        """query: [word1, word2, ..., wordn]"""
        scores = self.model.get_scores(query)
        rtn = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:topk]
        return rtn[0][0], rtn[1][0]
