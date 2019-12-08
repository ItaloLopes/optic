# -*- coding: utf-8 -*-

import random

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

# LSTM and FFNN for popularity

class wordLSTM(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, dropout, pad_idx, batch_size, weights_matrix):
        super(wordLSTM, self).__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_size = batch_size
        padding_idx = pad_idx

        # Setting each layer of the NN
        num_embeddings, embedding_dim = weights_matrix.shape
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.embedding.load_state_dict({'weight': weights_matrix})
        # Set requires_grad for embedding as False due to lack of memory 
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(self.emb_dim, self.hid_dim, self.n_layers, dropout=dropout, bidirectional=True, batch_first=True)
        #self.hidden2tag = nn.Linear( hid_dim*2, 1)
        self.dropout = nn.Dropout(dropout)
    
    def init_hidden(self):
        return (torch.zeros(self.n_layers*2, self.batch_size, self.hid_dim),
            torch.zeros(self.n_layers*2, self.batch_size, self.hid_dim))

    # sequence_idx = [batch_size, sequence_length]
    def forward(self, sequence_idx):
        #self.hidden = self.init_hidden(sequence_idx.shape[1])
        #sequence_idx.t() # [sequence_length, batch_size]        
        embedded = self.dropout(self.embedding(sequence_idx)) # embedded = [sequence_length, batch_size, embedding_size]
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs = [batch_size, sequence_length, hidden_dimension]
        # hidden = [n_layers, batch_size, hidden_dimension]
        # hidden[-1] = [batch_size, sequence_length]
        lstm_output = outputs[:, -1, :]
        #lstm_output = torch.cat( [lstm_output, rank], 1)
        #tag_space = self.hidden2tag(lstm_output)
        # tag_space = [n_layers, sequence_length, output_dim]
        #tag_scores = F.log_softmax(tag_space, dim=1)
        #tag_space = tag_space.squeeze()
        return lstm_output

class FFNN(nn.Module):
    def __init__(self, hid_dim):
        super(FFNN, self).__init__()
        self.hid_dim = hid_dim * 2

        self.fc1 = torch.nn.Linear(self.hid_dim + 1 , self.hid_dim + 1)
        self.relu = torch.nn.ReLU()
        self.hidden2tag = nn.Linear( self.hid_dim + 1, 1 )
    
    def forward(self, lstm_output, rank):
        data_input = torch.cat( [lstm_output, rank], 1 )
        hidden = self.fc1(data_input)
        relu = self.relu(hidden)
        tag_space = self.hidden2tag(relu)
        tag_space = tag_space.squeeze()
        return tag_space

class Optic(nn.Module):
    def __init__(self, lstm, ffnn, device):
        super(Hel, self).__init__()
        self.lstm = lstm
        self.ffnn = ffnn
        self.device = device
    
    def forward(self, sequence_idx, rank):
        lstm_output = self.lstm(sequence_idx)
        output = self.ffnn(lstm_output, rank)
        return output.to(self.device)

    