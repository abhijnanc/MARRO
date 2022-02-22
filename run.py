import os
import sys
import torch
import logging
import argparse
import numpy as np
from os.path import join
from pathlib import Path
from statistics import mean
from logger_file import fetch_logger

from model.Marro_model import *
from datetime import datetime
from prepare_data import *
from train import *


now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")



import random

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

import warnings

warnings.filterwarnings("ignore", category=Warning)


def main(root = "", country = "UK"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', default=False, type=bool,
                        help='Whether the model uses pretrained sentence embeddings or not')
    parser.add_argument('--data_path', default='{}/data/text/'.format(root),
                        type=str, help='Folder to store the annotated text files')
    parser.add_argument('--data_docs_original',
                        default='{}/dataset/IN-train-set/'.format(root),
                        type=str, help='original sentences and files')
    parser.add_argument('--save_path',
                        default='{}/saved/saved_temp/'.format(root), type=str,
                        help='Folder where predictions and models will be saved')
    parser.add_argument('--cat_path',
                        default='{}/in_categories.txt'.format(root), type=str,
                        help='Path to file containing category details')
    parser.add_argument('--dataset_size', default=150, type=int, help='Total no. of docs')
    parser.add_argument('--num_folds', default=5, type=int, help='No. of folds to divide the dataset into')
    parser.add_argument('--device', default='cuda', type=str, help='cuda / cpu')

    parser.add_argument('--use_marro', default=False, type=bool, help='use the tf embeddigns instead of the current sent2vec embeddings, add attention')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--print_every', default=1, type=int, help='Epoch interval after which validation macro f1 and loss will be printed')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning Rate')
    parser.add_argument('--min_lr', default=5 / 10000, type=float, help='min Learning Rate')
    parser.add_argument('--lr_decay_factor', default=0.95, type=float, help='Learning Rate Decay')
    parser.add_argument('--reg', default=0, type=float, help='L2 Regularization')
    
    parser.add_argument('--emb_dim', default=200, type=int, help='Sentence embedding dimension')
    parser.add_argument('--word_emb_dim', default= 200, type=int, help='Word embedding dimension, applicable only if pretrained = False')
    parser.add_argument('--epochs', default = 180, type=int)
    parser.add_argument('--val_fold', default='cross', type=str, help='Fold number to be used as validation, use cross for num_folds cross validation')
    parser.add_argument('--use_attention', default=False, type=bool)
    parser.add_argument('--attention_heads', default=5, type=int)
    parser.add_argument('--encoder_blocks', default=2, type=int)
    
    args = parser.parse_args()



    dir_name = 'experiment_logs/{}-{}-{}-{}-{}/'.format(country, args.batch_size, args.attention_heads, args.encoder_blocks, args.epochs)
    
    log_dir = join(root, dir_name)

    if not Path(log_dir).exists():
          Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger_filename = str(dt_string)+'test_logger.log'
    logger_filename_full = join(log_dir, logger_filename)
    
    if not os.path.exists(logger_filename_full):
      Path(logger_filename_full).touch()
    
    logging.basicConfig(filename=logger_filename_full, filemode='w', format='%(name)s - %(levelname)s - %(message)s')

    logger = fetch_logger('run file logger')
    logger.info("Starting the run for Rhetoric role classification")
    logger.info("test statement")


    logger.info('model is for : {}'.format(args.cat_path))
    logger.info('model is pretrained : {}'.format(args.pretrained))
    logger.info('usinf tf embedding plus attention model : {}'.format(args.use_marro))
    logger.info('attention heads used in model : {}'.format(args.attention_heads))
    logger.info('encoder blocks for attention : {}'.format(args.encoder_blocks))
    logger.info('training batch size : {}'.format(args.batch_size))
    logger.info('training learning rate : {}'.format(args.lr))
    logger.info('training epochs : {}'.format(args.epochs))

    print('\nPreparing data ...', end=' ')
    idx_order = prepare_folds(args)
    x, y, word2idx, tag2idx, original_sentences = prepare_data(idx_order, args)

    print('Vocabulary size:', len(word2idx))
    print('#Tags:', len(tag2idx))

    # Dump word2idx and tag2idx
    with open(args.save_path + 'word2idx.json', 'w') as fp:
        json.dump(word2idx, fp)
    with open(args.save_path + 'tag2idx.json', 'w') as fp:
        json.dump(tag2idx, fp)

    if args.val_fold == 'cross':

        print('\nCross-validation\n')
        avg_f1 = []
        attention_df = []
        for f in range(args.num_folds):
            print('\nInitializing model ...', end=' ')

            elif args.use_marro == True:
                model = MARRO(len(tag2idx), args.emb_dim, tag2idx['<start>'],
                              tag2idx['<end>'], tag2idx['<pad>'], vocab_size=len(word2idx),
                              word_emb_dim=args.word_emb_dim, pretrained=args.pretrained,
                              device=args.device, attention_heads=args.attention_heads,
                              num_blocks=args.encoder_blocks).to(args.device)

            print('Done')

            print('\nEvaluating on fold', f, '...')
            if f == 0 :
              logger.info("the model architecture is : \n {}".format(model))

            curr_fold_f1, _ = learn(model, x, y, tag2idx, f, args, idx_order,
                                                    original_docs=original_sentences)

            avg_f1.append(curr_fold_f1)

        print("average F1 across folds is : " + str(mean(avg_f1)))



if __name__ == '__main__':
    root  = os.getcwd()
    root = os.path.join(root, '.')
    main(root, "India")
