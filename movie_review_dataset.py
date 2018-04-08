# -*- coding: utf-8 -*-

import csv
import os

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from nsml import GPU_NUM


csv.field_size_limit(500 * 1024 * 1024 * 10000)
train_data_name = 'train_data'
train_label_name = 'train_label'
test_data_name = 'test_data'


class MovieReviewDataset(Dataset):
    def __init__(self, dataset_path):
        data_path = os.path.join(dataset_path, 'train', train_data_name)
        data_label = os.path.join(dataset_path, 'train', train_label_name)

        print("Data path: ", data_path)
        self.loaded_data = []

        with open(data_path, 'rt', 50 * 1024 * 1024) as data_csvfile,\
                open(data_label, 'rt') as label_csv:  # maximum buffer size == 50 MB
            # the fastest way of counting the number of total file lines.
            self.total_data_length = sum(1 for _ in data_csvfile)
            print('Total data length: ', self.total_data_length)

            # reset file pointer to 0
            data_csvfile.seek(0)
            label_csv.seek(0)
            read_line = csv.reader(data_csvfile)
            read_label = csv.reader(label_csv)

            for idx, (data, label) in enumerate(zip(read_line, read_label)):
                try:
                    data = data[0]
                    label = int(label[0])
                    self.loaded_data.append([data, label])
                except:
                    print(data, label, idx, data_path)
                    continue

    def __len__(self):
        return self.total_data_length

    def __getitem__(self, idx):
        # this should return only one pair of instance (x, y).
        seqs = self.loaded_data[idx]
        batch_input = seqs[0]
        batch_target = seqs[1]
        return [batch_input, batch_target]


def cusmtom_collate_fn(data):
    """Creates mini-batch tensors from the list of list [batch_input, batch_target].
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
        data: list of list [batch_input, batch_target].

    Refs:
        Thanks to yunjey. (https://github.com/yunjey/seq2seq-dataloader/blob/master/data_loader.py)
    """

    # data sampler returns an indices to select into mini-batch.
    # than the collate_function gather the selected data.
    # in this example, we use pack_padded_sequence so always ordering with decsending
    # so the targets should be ordered samely here.

    # sort the sequences as descending order to use 'pack_padded_sequence' before loading.
    data.sort(key=lambda x: len(x[0]), reverse=True)
    tensor_targets = torch.LongTensor([pair[1] for pair in data])
    return [data, tensor_targets]


def load_batch_input_to_memory(batch_input, has_targets=True):
    if has_targets:
        batch_input = [[int(i) for i in seq[0]] for seq in batch_input]
    else:
        batch_input = batch_input

    # Get the length of each seq in your batch
    tensor_lengths = torch.LongTensor(list(map(len, batch_input)))
    # Zero-padded long-Matirx size of (B, T)
    tensor_seqs = torch.zeros((len(batch_input), tensor_lengths.max())).long()
    for idx, (seq, seqlen) in enumerate(zip(batch_input, tensor_lengths)):
        tensor_seqs[idx, :seqlen] = torch.LongTensor(seq)

    return [tensor_seqs, tensor_lengths]


def get_dataloaders(dataset_path, type_of_data, batch_size, ratio_of_validation=0.2, shuffle=True):
    if type_of_data == 'train':
        num_workers = 0  # num of threads to load data, default is 0. if you use thread(>1), don't confuse even if debug messages are reported asynchronously.
        train_movie_dataset = MovieReviewDataset(dataset_path=dataset_path)

        num_train = len(train_movie_dataset)
        split_point = int(ratio_of_validation*num_train)

        indices = list(range(num_train))
        if shuffle:
            np.random.shuffle(indices)

        train_idx, val_idx = indices[split_point:], indices[: split_point]
        train_sampler = SubsetRandomSampler(train_idx)  # Random sampling at every epoch without replacement in given indices
        val_sampler = SubsetRandomSampler(val_idx)

        # 'pin_memory=True' allows for you to use fast memory buffer with way of calling '.cuda(async=True)' function.
        pin_memory = True if GPU_NUM else False

        train_loader = torch.utils.data.DataLoader(
            train_movie_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
            pin_memory=pin_memory, collate_fn=cusmtom_collate_fn)
        val_loader = torch.utils.data.DataLoader(
            train_movie_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers,
            pin_memory=pin_memory, collate_fn=cusmtom_collate_fn)
        return [train_loader, val_loader]

    else:
        raise BaseException('This function is only for train data set')


def read_test_file(dataset_path):
    data_path = os.path.join(dataset_path, 'test', test_data_name)  # Test data only
    loaded_data = []
    with open(data_path, 'rt') as data_csvfile:
        read_line = csv.reader(data_csvfile)
        for idx, data in enumerate(read_line):
            data = data[0]
            loaded_data.append([data])
    return loaded_data
