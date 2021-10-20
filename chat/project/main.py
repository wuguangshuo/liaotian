
import argparse
from utils import get_corpus, word_tokenize, build_word_embedding
from model_zoo.bm25_model import BM25RetrievalModel
from model_zoo.tfidf_model import TFIDFRetrievalModel
from model_zoo.sif_model import SIFRetrievalModel
from model_zoo.WMD_model import WMDRetrievalModel

parser = argparse.ArgumentParser(description='Information retrieval model hyper-parameter setting.')
parser.add_argument('--input_file_path', type=str, default='./ChangCheng.xls', help='Training data location.')
parser.add_argument('--model_type', type=str, default='wmd', help='Different information retrieval models.')
parser.add_argument('--gensim_model_path', type=str, default='./cached/gensim_model.pkl')
parser.add_argument('--pretrained_gensim_embddings_file', type=str, default='./cached/gensim_word_embddings.pkl')
parser.add_argument('--cached_gensim_embedding_file', type=str, default='./cached/embeddings_gensim.pkl')
parser.add_argument('--embedding_dim', type=int, default=100)
parser.add_argument('--max_seq_len', type=int, default=30)


args = parser.parse_args()

# 读取 问题-答案 数据
questions_src, answers = get_corpus(args.input_file_path)

# 分词
questions = [word_tokenize(line) for line in questions_src]
answers_corpus = [word_tokenize(line) for line in answers]

# 第一次运行，需要训练词向量
# print('\nBuild gensim model and word vectors...')
# build_word_embedding(questions+answers_corpus, args.gensim_model_path, args.pretrained_gensim_embddings_file)


def predict(model, query):
    """softmax each model predictions and summarize"""
    query = word_tokenize(query)
    res_all = []
    top_1, top_2 = model.get_top_similarities(query, topk=2)
    return questions_src[top_1], answers[top_1], questions_src[top_2], answers[top_2]


if __name__ == '__main__':

    query = '如何审批假期'
    if args.model_type == 'bm25':
        bm25_model = BM25RetrievalModel(questions)
        res = predict(bm25_model, query)
    elif args.model_type == 'tfidf':
        tfidf_model = TFIDFRetrievalModel(questions)
        res = predict(tfidf_model, query)
    elif args.model_type == 'sif':
        sif_model = SIFRetrievalModel(questions, args.pretrained_gensim_embddings_file,
                                      args.cached_gensim_embedding_file, args.embedding_dim)
        res = predict(sif_model, query)
    elif args.model_type == 'wmd':
        wmd_model = WMDRetrievalModel(questions, args.gensim_model_path)
        res = predict(wmd_model, query)
    else:
        raise ValueError('Invalid model type!')

    print('Query: ', query)
    print('\nQuestion 1: ', res[0])
    print('Answer 1: ', res[1])
    print('\nQuestion 2: ', res[2])
    print('Answer 2: ', res[3])

