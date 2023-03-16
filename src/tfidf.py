from haystack.nodes import TfidfRetriever
from haystack.document_stores import InMemoryDocumentStore
from utils import *


def get_tfidf_predictions(data, val_passages):
    document_store_tfidf = InMemoryDocumentStore()
    document_store_tfidf.write_documents(prepare_dev_data(val_passages))
    TFIDFretriever = TfidfRetriever(document_store_tfidf)
    return get_retrieved_docs(data, TFIDFretriever)
