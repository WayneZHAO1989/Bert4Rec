import os
import random
import numpy as np
import pickle
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, Dataset
from abc import ABCMeta



class RandomNegativeSampler(metaclass=ABCMeta):
    def __init__(self, train, val, test, user_count, item_count, sample_size, seed, save_folder):
        self.train = train
        self.val = val
        self.test = test
        self.user_count = user_count
        self.item_count = item_count
        self.sample_size = sample_size
        self.seed = seed
        self.save_folder = save_folder

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        negative_samples = {}
        print('Sampling negative items')
        for user in trange(self.user_count):
            if isinstance(self.train[user][1], tuple):
                seen = set(x[0] for x in self.train[user])
                seen.update(x[0] for x in self.val[user])
                seen.update(x[0] for x in self.test[user])
            else:
                seen = set(self.train[user])
                seen.update(self.val[user])
                seen.update(self.test[user])

            samples = []
            for _ in range(self.sample_size):
                item = np.random.choice(self.item_count) + 1
                while item in seen or item in samples:
                    item = np.random.choice(self.item_count) + 1
                samples.append(item)

            negative_samples[user] = samples

        return negative_samples

    def get_negative_samples(self):
        savefile_path = 'random_negative_samplesize{}_seed{}.pkl'.format(self.sample_size, self.seed)
        print("negative samples path:", savefile_path)
        if os.path.exists(savefile_path):
            print('Negatives samples exist. Loading.')
            negative_samples = pickle.load(open(savefile_path, 'rb'))
        else:
            print("Negative samples don't exist. Generating.")
            negative_samples = self.generate_negative_samples()
            with open(savefile_path, 'wb') as f:
                pickle.dump(negative_samples, f)
        return negative_samples



class BertDataloader(metaclass=ABCMeta):
    def __init__(self, args, umap, smap, train, val, test, save_folder):
        super().__init__()
        
        self.args = args
        self.rng = random.Random(12345)
        
        self.train = train
        self.val = val
        self.test = test
        self.umap = umap
        self.smap = smap
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)
        
        self.save_folder = save_folder
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.item_count + 1

        train_negative_sampler = RandomNegativeSampler(self.train, self.val, self.test,
                                                          self.user_count, self.item_count,
                                                          args.train_negative_sample_size,
                                                          args.train_negative_sampling_seed,
                                                          self.save_folder)

        test_negative_sampler = RandomNegativeSampler(self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         self.save_folder)

        self.train_negative_samples = train_negative_sampler.get_negative_samples()
        self.test_negative_samples = test_negative_sampler.get_negative_samples()

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_eval_loader(mode='val')
        test_loader = self._get_eval_loader(mode='test')
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = BertTrainDataset(self.train, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True)
        return dataloader

    def _get_eval_loader(self, mode):
        answers = self.val if mode == 'val' else self.test
        dataset = BertEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True)
        return dataloader

class BertTrainDataset(Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, item_count, rng):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.item_count = item_count
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)

        tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.item_count))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        tokens = [0] * (self.max_len - len(tokens)) + tokens
        labels = [0] * (self.max_len - len(tokens)) + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]

class BertEvalDataset(Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)
