# -*- coding: utf-8 -*-

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from nsml import DATASET_PATH, IS_DATASET, GPU_NUM
import nsml

from korean_character_parser import decompose_str_as_one_hot
from movie_review_dataset import load_batch_input_to_memory, get_dataloaders, read_test_file


def bind_model(model):
    def save(filename, *args):
        # save the model with 'checkpoint' dictionary.
        checkpoint = {
            'model': model.state_dict()
        }
        torch.save(checkpoint, filename)

    def load(filename, *args):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        print('Model loaded')

    def infer(raw_data, **kwargs):
        data = raw_data['data']
        model.eval()
        output_prediction = model(data)
        point = output_prediction.data.squeeze(dim=1).tolist()
        return list(zip(point, point))

    # function in function is just used to divide the namespace.
    nsml.bind(save, load, infer)


def data_loader(dataset_path, train=False, batch_size=200, ratio_of_validation=0.1, shuffle=True):
    if train:
        return get_dataloaders(dataset_path=dataset_path, type_of_data='train',
                               batch_size=batch_size, ratio_of_validation=ratio_of_validation,
                               shuffle=shuffle)
    else:
        data_dict = {'data': read_test_file(dataset_path=DATASET_PATH)}
        return data_dict


class LSTM_Regression(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, character_size, output_dim, minibatch_size):
        super(LSTM_Regression, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.character_size = character_size
        self.output_dim = output_dim
        self.minibatch_size = minibatch_size

        # this embedding is a table to handle sparse matrix instead of one-hot coding. so we just feed a list of indexes.
        self.embeddings = nn.Embedding(self.character_size, self.embedding_dim)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers)

        # non-linear function is defined later.
        self.hidden2score = nn.Linear(self.hidden_dim, self.output_dim)

    def init_hidden(self):
        # there are 2 groups of hiddens. first one is hidden state, second one is cell state.
        # if bidirectional, there are actually doubled hidden units of LSTM
        initializer_1 = autograd.Variable(torch.zeros(self.num_layers, self.minibatch_size, self.hidden_dim))
        initializer_2 = autograd.Variable(torch.zeros(self.num_layers, self.minibatch_size, self.hidden_dim))

        if GPU_NUM:
            return initializer_1.cuda(), initializer_2.cuda()
        else:
            return initializer_1, initializer_2

    def forward(self, data):
        # correct data format of input. list of list
        # [ ['전편보단 못했지만 그럭저럭 재밌었던 영화였음', 2] ]

        preprocessed = [decompose_str_as_one_hot(datum[0], warning=False) for datum in data]
        preprocessed.sort(key=lambda x: len(x), reverse=True)

        var_seqs, var_lengths = load_batch_input_to_memory(preprocessed, has_targets=False)

        var_seqs = autograd.Variable(var_seqs)
        var_lengths = autograd.Variable(var_lengths)
        if GPU_NUM:
            var_seqs = var_seqs.cuda(async=True)
            var_lengths = var_lengths.cuda(async=True)

        var_seqs = var_seqs[:, :var_lengths.data.max()]
        self.minibatch_size = len(var_lengths)

        # Zero padded maxtrix shaped (Batch, Time) ->  Tensor shaped (Batch, Time, Embeded_Feature)
        embeds = self.embeddings(var_seqs)

        # (Batch X Compact_Time , Embeded_Feature)
        packed_x = pack_padded_sequence(embeds, var_lengths.data.cpu().numpy(), batch_first=True)
        # Compact_time means a tensor without pads. So this is a concatenated tensor with only useful sequenses.
        # Ex) [[53, 16], [40,16]] --> [53+40, 16]

        # This makes the memory the parameters and its grads occupied contiguous for efficiency of memory usage..
        self.lstm.flatten_parameters()

        # _hidden is not important, the output is important.
        packed_output, _hidden = self.lstm(packed_x, self.init_hidden())

        # Reverse operation of pack_padded_sequence. as (Time, Batch, Concatenation of 2 directional hidden's output).
        lstm_outs, _ = pad_packed_sequence(packed_output)

        # Implementation of last relevant output indexing.
        if GPU_NUM:
            idx = ((var_lengths - 1).view(-1, 1).expand(lstm_outs.size(1), lstm_outs.size(2)).unsqueeze(
                0)).cuda()  # async=True
        else:
            idx = ((var_lengths - 1).view(-1, 1).expand(lstm_outs.size(1),
                                                        lstm_outs.size(2)).unsqueeze(0))
        # squeeze remove all ones, so it breaks when batch size is 1. dim=0 should be added to avoid it
        last_lstm_outs = lstm_outs.gather(0, idx).squeeze(dim=0)
        output_activation = self.hidden2score(last_lstm_outs)

        # 1-10 스케일로 변환
        output_pred = (F.sigmoid(output_activation)*9)+1
        return output_pred


def inference_loop(data_loader, model, loss_function, optimizer, learning = True): # , without_training=False
    if learning:
        model.train()  # select train mode
    else:
        model.eval()

    sum_loss = 0.0
    num_of_instances = 0
    for i, (data, label) in enumerate(data_loader):
        # we need to clear out the hidden state of the LSTM, detaching it from its history on the last instance.
        # model.hidden = model.init_hidden() # but this may be skipped if we use lstm(X, self.init_hidden())

        # Tensors not supported in DataParallel. You should put Variable to use data_parallel before call forward().
        if GPU_NUM:
            var_targets = autograd.Variable(label.float(), requires_grad=False).cuda(async=True)
        else:
            var_targets = autograd.Variable(label.float(), requires_grad=False)

        output_predictions = model(data)

        loss = loss_function(output_predictions, var_targets)
        sum_loss += loss.data[0]*len(label)  # Sum of loss = (MSE * num_of_instance)
        num_of_instances += len(label)

        if learning:
            # Remember that Pytorch accumulates gradients. So we need to clear them out before each instance
            optimizer.zero_grad() # model.zero_grad(). this is also same if optimizer = optim.Optimizer(model.parameters())
            loss.backward()
            optimizer.step()
        print('Batch : ', i+1, '/', len(data_loader), ', MSE in this minibatch: ', loss.data[0])

    return sum_loss/num_of_instances  # retrun mean loss


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--epochs', type=int, default=300)  # 300
    args.add_argument('--batch', type=int, default=1000)  # 1000
    args.add_argument('--embedding', type=int, default=8)  # 8
    args.add_argument('--hidden', type=int, default=512)  # 512
    args.add_argument('--layers', type=int, default=2)  # 2
    args.add_argument('--initial_lr', type=float, default=0.001)  # default : 0.001 (initial learning rate)
    args.add_argument('--char', type=int, default=251)  # Do not change this
    args.add_argument('--output', type=int, default=1)  # Do not change this
    args.add_argument('--mode', type=str, default='train')  # 'train' or 'test' (for nsml)
    args.add_argument('--pause', type=int, default=0)  # Do not change this (for nsml)
    args.add_argument('--iteration', type=str, default='0')  # Do not change this (for nsml)
    config = args.parse_args()

    initial_time = time.time()
    random_seed = 1234
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    model = LSTM_Regression(config.embedding, config.hidden, config.layers,
                            config.char, config.output, config.batch)
    loss_function = nn.MSELoss()
    if GPU_NUM:
        model = model.cuda()
        loss_function = nn.MSELoss().cuda()

    optimizer = optim.Adam(model.parameters(), lr=config.initial_lr)
    adjust_learning_rate = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    bind_model(model)

    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        if IS_DATASET:
            train_loader, val_loader = data_loader(dataset_path=DATASET_PATH, train=True,
                                                   batch_size=config.batch,
                                                   ratio_of_validation=0.1, shuffle=True)
        else:
            data_path = '../dummy/movie_review/'  # NOTE: load from local PC
            train_loader, val_loader = data_loader(dataset_path=data_path, train=True,
                                                   batch_size=config.batch,
                                                   ratio_of_validation=0.1, shuffle=True)
        dataloader_initialize = time.time()
        min_val_loss = np.inf
        for epoch in range(config.epochs):
            # train on train set
            train_loss = inference_loop(train_loader, model, loss_function, optimizer, learning=True)
            # evaluate on validation set
            val_loss = inference_loop(val_loader, model, loss_function, None, learning=False)

            # if you want, you can apply the learning rate decaying.
            adjust_learning_rate.step(val_loss)

            print('epoch:', epoch, ' train_loss:', train_loss, ' val_loss:', val_loss, ' min_val_loss:', min_val_loss)
            nsml.report(summary=True, scope=locals(), epoch=epoch, total_epoch=config.epochs,
                        train__loss=train_loss, val__loss=val_loss, min_val_loss=min_val_loss)

            if val_loss <= min_val_loss:
                min_val_loss = val_loss
                nsml.save(epoch)
            else:  # default save
                if epoch % 30 == 0:
                    nsml.save(epoch)
        final_time = time.time()
