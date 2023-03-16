import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


def prepare_dev_data(dataset):
    val_passages_dict = dataset.to_dict('records')
    document_store_data = []
    for passage in val_passages_dict:
        document_store_data.append(
            {'content': passage['context'], 'meta': {'name': passage['title']}}
        )
    return document_store_data


def get_retrieved_docs(dataset, retriever):
    if type(dataset) == str:
        output= []
        retrieved_list = retriever.retrieve(dataset, top_k=10)
        for doc in retrieved_list:
            output.append(doc.to_dict()['content'])
    else:
        output = []
        for query in tqdm(dataset.question):
            retrieved = retriever.retrieve(query, top_k=10)
            retrieved_list = []
            for doc in retrieved:
                retrieved_list.append(doc.to_dict()['content'])
            output.append(retrieved_list)
    return output


def compute_accuracy(data, k=1, model='DPR_retrieval'):
    count = 0
    for context, retrieved in zip(data.context, data[model]):
        if context in retrieved[:k]:
            count += 1
    return count/len(data)


def compute_performance(data, iter=9):
    performance = {}
    for model in ['bm25', 'DPR_retrieval', 'TFIDF']:
        tmp = []
        for k in range(10):
            tmp.append(compute_accuracy(data, k=k + 1, model=model))
        performance[model] = tmp
    for model in ['bm25', 'DPR_retrieval', 'TFIDF']:
        print("Accuracy at {} with {}: {}".format(iter, model, performance[model][iter]))
    sns.lineplot(data=pd.DataFrame(performance))
    plt.xlabel('k')
    plt.ylabel('Accuracy@k')
    plt.title('Comparison accuracy@k')
    plt.savefig('../Results/accuracy_at_k.png')

