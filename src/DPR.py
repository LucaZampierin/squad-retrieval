from haystack.nodes import DensePassageRetriever
from haystack.utils import fetch_archive_from_http
from haystack.document_stores import InMemoryDocumentStore, ElasticsearchDocumentStore
# from haystack.document_stores.faiss import FAISSDocumentStore
from utils import *


save_dir = "../DPR"


def get_dpr_predictions(data, val_passages):
    # This function will give an error on Windows as the faiss library is not well supported.
    # document_store = FAISSDocumentStore()
    # document_store = ElasticsearchDocumentStore()
    document_store.write_documents(prepare_dev_data(val_passages))
    DPR_retriever = DensePassageRetriever.load(load_dir=save_dir, document_store=document_store)
    document_store.update_embeddings(DPR_retriever)
    return get_retrieved_docs(data, DPR_retriever)

