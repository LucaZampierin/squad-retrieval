from bm25 import get_bm25_predictions
from tfidf import get_tfidf_predictions
from DPR import get_dpr_predictions
from create_DPR_data import load_DPR_data
from read_squad_data import split_train_dev
import argparse
from finetune_DPR import train_DPR
from utils import *
import os
import json
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to evaluate the IR system")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'],
                        help='Set train if the DPR should be further finetuned', default='eval')
    args = vars(parser.parse_args())

    if not os.listdir('../Data/train_dev_split'):
        print("Splitting the SQUAD data into train and development")
        split_train_dev()

    with open('../Data/train_dev_split/dev_queries.json', 'r') as f:
        val_data = pd.DataFrame(json.load(f))

    with open('../Data/train_dev_split/dev_passages.json', 'r') as f:
        val_passages = pd.DataFrame(json.load(f))

    if args['mode'] == 'train':
        if not os.listdir('../Data/DPRdata'):
            print("Building the dataset to train the DPR model")
            load_DPR_data('train')
            load_DPR_data('dev')
        train_DPR()

    if not os.listdir('../Results'):
        print("Evaluating the systems")
        # The DPR will give an error on windows as the faiss library is not well supported
        val_data['DPR'] = get_dpr_predictions(val_data, val_passages)
        val_data['bm25'] = get_bm25_predictions(val_data, val_passages, k=10)
        val_data['TFIDF'] = get_tfidf_predictions(val_data, val_passages)
        val_data.to_json('../Results/dev_with_results.json', orient='records')
    else:
        with open('../Results/dev_with_results.json', 'r') as f:
            val_data = pd.DataFrame(json.load(f))

    print(val_data.columns)
    compute_performance(val_data)



