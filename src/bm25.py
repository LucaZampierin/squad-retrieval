from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import string
import numpy as np
from rank_bm25 import BM25Okapi
nltk.download('stopwords')
nltk.download('punkt')

stemmer = PorterStemmer()


def tokenize(text):
    """
    It tokenizes and stems an input text.

    :param text: str, with the input text
    :return: list, of the tokenized and stemmed tokens.
    """
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    return " ".join([stemmer.stem(word.lower()) for word in tokens])


def vectorize_query(query):
    q = query.split()
    q = [stemmer.stem(w) for w in q]
    return q


def tokenize_docs(passages):
    original_documents = [x.strip() for x in passages.context]
    documents = [tokenize(d).split() for d in original_documents]
    context_index = range(len(original_documents))
    return original_documents, documents, context_index


def get_bm25_predictions(data, passages, k=10):
    original_docs, tokenized_docs, context_idx = tokenize_docs(passages)
    bm25 = BM25Okapi(tokenized_docs)
    if type(data) == str:
        tokenized_query = vectorize_query(data)
        output = bm25.get_top_n(tokenized_query, original_docs, n=k)

    else:
        output = []
        for query in tqdm(data.question):
            tokenized_query = vectorize_query(query)
            retrieved = bm25.get_top_n(tokenized_query, original_docs, n=k)
            output.append(retrieved)
    return output


