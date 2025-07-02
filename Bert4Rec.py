#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import random
import json
from collections import deque
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')


from config import args
from dataloader import BertDataloader, BertTrainDataset, BertEvalDataset
from utils import recalls_and_ndcgs_for_ks, AverageMeterSet
from model import BERTModel



## Dataloader
def load_data(filePath, min_rating=0, min_sc=0, min_uc=5):
    """M1-1M"""
    
    df = pd.read_csv(filePath, sep='::', header=None)
    df.columns = ['uid', 'sid', 'rating', 'timestamp']
    
    # Turning into implicit ratings
    df = df[df['rating'] >= min_rating]
    
    # Filtering triplets
    if min_sc > 0:
        item_sizes = df.groupby('sid').size()
        good_items = item_sizes.index[item_sizes >= min_sc]
        df = df[df['sid'].isin(good_items)]
    
    if min_uc > 0:
        user_sizes = df.groupby('uid').size()
        good_users = user_sizes.index[user_sizes >= min_uc]
        df = df[df['uid'].isin(good_users)]
    
    # Densifying index
    umap = {u: i for i, u in enumerate(set(df['uid']))}
    smap = {s: i for i, s in enumerate(set(df['sid']))}
    df['uid'] = df['uid'].map(umap)
    df['sid'] = df['sid'].map(smap)
    
    # Split into train/val/test
    user_group = df.groupby('uid')
    user2items = user_group.apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
    train, val, test = {}, {}, {}
    for user in range(len(umap)):
        items = user2items[user]
        train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
    
    return df, umap, smap, train, val, test





## Evaluation

def calculate_loss(batch):
    seqs, labels = batch
    logits = model(seqs)  # B x T x V
    logits = logits.view(-1, logits.size(-1))  # (B*T) x V
    labels = labels.view(-1)  # B*T
    loss = ce(logits, labels)
    return loss

def calculate_metrics(batch, metric_ks=args.metric_ks):
    seqs, candidates, labels = batch
    scores = model(seqs)  # B x T x V
    scores = scores[:, -1, :]  # B x V
    scores = scores.gather(1, candidates)  # B x C

    metrics = recalls_and_ndcgs_for_ks(scores, labels, metric_ks)
    return metrics





## load data
_, umap, smap, train, val, test = load_data(args.data_path, args.min_rating, args.min_sc, args.min_uc)

args.num_items = len(smap)


dataloader = BertDataloader(args, umap, smap, train, val, test, './')
train_dataloader, val_dataloader, test_dataloader = dataloader.get_pytorch_dataloaders()



## model init
model = BERTModel(args)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
ce = nn.CrossEntropyLoss(ignore_index=0)




def run_eval(mode):
    model.eval()

    average_meter_set = AverageMeterSet()

    with torch.no_grad():
        if mode=='val':
            tqdm_dataloader = tqdm(val_dataloader)
        else:
            tqdm_dataloader = tqdm(test_dataloader)
            
        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(args.device) for x in batch]
            metrics = calculate_metrics(batch)

            for k, v in metrics.items():
                average_meter_set.update(k, v)
            description_metrics = ['NDCG@%d' % k for k in args.metric_ks[:3]] +\
                                  ['Recall@%d' % k for k in args.metric_ks[:3]]
            description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
            description = description.format(*(average_meter_set[k].avg for k in description_metrics))
            tqdm_dataloader.set_description(description)
            


def train_one_epoch( epoch, accum_iter):

    model.train()
    lr_scheduler.step()
    loss_history = deque(maxlen=200)
    
    for batch_idx, batch in enumerate(tqdm(train_dataloader, dynamic_ncols=True) ):
        batch_size = batch[0].size(0)
        batch = [x.to(args.device) for x in batch]
    
        optimizer.zero_grad()
        loss = calculate_loss(batch)
        loss.backward()
    
        optimizer.step()
    
        loss_history.append(loss.tolist())

    torch.save(model, "model_checkpoint_epoch_{}.pt".format( str(epoch).zfill(2) ))
    print("Epoch: ", epoch, ", Loss:", np.mean(loss_history))

    return accum_iter


accum_iter = 0
run_eval(mode='val')

for epoch in range(args.num_epochs):
    accum_iter = train_one_epoch(epoch, accum_iter)
    run_eval(mode='val')

run_eval(mode='test')