import pandas as pd
import json


PATH = '../Data/SQuAD-v1.1.json'


def split_train_dev():
    with open(PATH, 'r') as f:
        squad_data = json.load(f)

    contexts = list()
    queries = list()

    for j, item in enumerate(squad_data[1]):
      if j > 0:
        context_id = j-1
        role = item[4][2]
        context = item[1][2]
        title = item[3][2]
        contexts.append({'title': title, 'context': context, 'role': role})
        for i, question_answer in enumerate(item[2][2]):
          if i > 0:
            question = question_answer[1][2]
            answer = question_answer[2][2][1]
            idx = question_answer[3][2][1]
            qst_id = question_answer[4][2]
            queries.append({'question': question, 'answer': answer, 'answer_idx': idx,
                            'question_ID': qst_id, 'role': role, 'context_id': context_id,
                            'title_context': title, 'context': context})
    questions = pd.DataFrame(queries)
    passages = pd.DataFrame(contexts)
    train_queries = questions[questions['role'] == "'Training'"]
    val_queries = questions[questions['role'] == "'Validation'"]
    train_passages = passages[passages['role'] == "'Training'"]
    val_passages = passages[passages['role'] == "'Validation'"]

    train_queries.to_json('../Data/train_dev_split/train_queries.json', orient='records')
    val_queries.to_json('../Data/train_dev_split/dev_queries.json', orient='records')
    train_passages.to_json('../Data/train_dev_split/train_passages.json', orient='records')
    val_passages.to_json('../Data/train_dev_split/dev_passages.json', orient='records')
