from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity

class WMDRetrievalModel:
    def __init__(self,corpus,gensiom_model_path):
        Word2Vec_model=Word2Vec.load(gensiom_model_path)
        self.wmd_similarity=WmdSimilarity(corpus,Word2Vec_model,2)
    def get_top_similarities(self, query, topk=10):
        sims=self.wmd_similarity[query]

        return sims[0][0], sims[1][0]