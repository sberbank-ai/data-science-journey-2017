#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
"""A script to run the DrQA reader model interactively for russian language."""

import sys
# add path
sys.path.append('.')

import torch
import argparse
import logging
import csv

import os
import pandas as pd
import tqdm

from drqa.reader import Predictor

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# ------------------------------------------------------------------------------
# Commandline arguments & init
# ------------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None,
                    help='Path to model to use')
parser.add_argument('--tokenizer', type=str, default=None,
                    help=("String option specifying tokenizer type to use "
                          "(e.g. 'SimpleTokenizer (it is hardcode)')"))
parser.add_argument('--no-cuda', action='store_true',
                    help='Use CPU only')
parser.add_argument('--gpu', type=int, default=-1,
                    help='Specify GPU device id to use')
parser.add_argument('--no-normalize', action='store_true',
                    help='Do not softmax normalize output scores.')
parser.add_argument('--embedding_file', action='store',
                    help='word2vec_file')
parser.add_argument('--batch_size', action='store',
                    help='batch size in pred', default=100)
parser.add_argument('--num_workers', action='store',
                    help='num workers, zero is disable multiprocessing', default=0)
args = parser.parse_args()
logging.info(str(args))

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    logger.info('Running on CPU only.')

# load model
predictor = Predictor(args.model, args.tokenizer, num_workers=int(args.num_workers), normalize=not args.no_normalize, embedding_file=args.embedding_file)

if args.cuda:
    predictor.cuda()


# help function for split predictions on chunks
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


DATA_FILE = os.environ.get('INPUT', 'train.csv')
PREDICTION_FILE = os.environ.get('OUTPUT', 'data/result.csv')

df = pd.DataFrame.from_csv(DATA_FILE, sep=',', index_col=None)
df = df[['paragraph_id', 'question_id', 'paragraph', 'question']]

result = []
for i in tqdm.tqdm(chunks(list(range(df.shape[0])), int(args.batch_size))):
    paragraph = df.paragraph.iloc[i].values
    question = df.question.iloc[i].values
    pred = predictor.predict_batch(zip(paragraph, question))
    # pred = predictor.predict(paragraph, question, None, 1)
    result.extend(map(lambda x: x[0][0], pred))

df['answer'] = result

df[["paragraph_id", "question_id", "answer"]].to_csv(PREDICTION_FILE, header=True, quoting=csv.QUOTE_NONNUMERIC, index=False)
