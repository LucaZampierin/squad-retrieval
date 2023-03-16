import numpy as np
from tqdm import tqdm
import json


def get_negative_ctx(query):
  for neg_context in query['trial']:
    if neg_context != query['context']:
      return [{"title": '', "text": neg_context, "passage_id": ''}]  


def load_dict(queries):
  for query_str in tqdm(queries):
    query = dict(query_str)
    if query['role'] == "'Training'":
      out_dict = {
                      "question": query['question'],
                      "answers": query['answer'],
                      "positive_ctxs": [{"title": query['title_context'], "text": query['context'], "passage_id": ""}],
                      "negative_ctxs": [],
                      "hard_negative_ctxs": get_negative_ctx(query),
                  }
      yield out_dict
    else:
      out_dict = {
                      "question": query['question'],
                      "answers": query['answer'],
                      "positive_ctxs": [{"title": query['title_context'], "text": query['context'], "passage_id": ""}],
                      "negative_ctxs": [],
                      "hard_negative_ctxs": [],
                  }
      yield out_dict


def load_DPR_data(phase):
    with open ('../Data/train_dev_split/{}_queries.json'.format(phase), 'r') as f:
      data = json.load(f)
    DPR_data = list(load_dict(data))
    with open('../Data/DPRdata/{}_DPR.json'.format(phase), "w", encoding="utf-8") as outfile:
        json.dump(list(DPR_data), outfile, indent=4, ensure_ascii=False)








  