# -*- coding: utf-8 -*-
# Script for training a Neural Network for the framework Oya


import time
import random
import math
import time
import pickle
import json
import logging 
import argparse

from NN import *

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import bcolz
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def generate_loss_graph(loss_train, loss_val, title, label_x, label_y):
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title, wrap=True)
    
    plt.plot(loss_train, label='Training Loss', color='red')
    plt.plot(loss_val, label='Valid Loss', color='green')

    return plt

def generate_auc_graph(auc, title, label_x, label_y):
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title, wrap=True)

    plt.plot(auc, label='AUC score', color='red')
    return plt

def save_plot(plot, output_path, format='svg', dpi=300, transparent=True):
    plt.savefig(output_path, format=format, dpi=dpi, transparent=transparent)

# Implementing the pytorch Dataset class for the framework 
class DatasetOya(Dataset):
    def __init__(self, file):
        self.data = file

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = torch.tensor(self.data[idx]['source'], dtype=torch.long)
        label = torch.tensor(self.data[idx]['target'], dtype=torch.float)
        rank = torch.tensor(self.data[idx]['rank'], dtype=torch.float)
        return text, label, rank

# Pad all instance for the same length
def normalize_dataset(dataset):
    max_len = 0
    # Get length of biggest instance
    for value in dataset:
        if max_len < len(value['source']):
            max_len = len(value['source'])
    
    # Pad smaller instances 
    for value in dataset:
        diff_len = max_len - len(value['source'])
        for i in range(diff_len):
            value['source'].append('<pad>')
    return dataset

#Trim the dataset acoording to the window size
def trim_dataset(dataset, ws):
    for value in dataset:
        trimmed_value = []
        left_ws = ws
        right_ws = 1
        entity = [w for w in value['source'] if w.startswith('dbr:')]
        entity_idx = value['source'].index(entity[0])
        # Add left context
        for i in range(ws):
            idx = entity_idx - left_ws
            if idx >= 0:
                trimmed_value.append(value['source'][idx])
            left_ws = left_ws - 1
        # Add entity
        trimmed_value.append(value['source'][entity_idx])
        # Add right context
        for i in range(ws):
            idx = entity_idx + right_ws
            if idx < len(value['source']):
                trimmed_value.append(value['source'][idx])
            right_ws = right_ws + 1
        value['source'] = trimmed_value
    return dataset


# Convert dataset to indexes that point to their respective embeddings in the weight_matrix
def convert_dataset(dataset, vocab):
    dataset = normalize_dataset(dataset)
    converted_dataset = []
    for value in dataset:
        instance = dict()
        temp_list = []
        temp_target = []
        temp_rank = []
        for w in value['source']:
            w = w.lower()
            try:
                # Sometimes, during the preprocessing of the tweet, the Tweet NLP parses incorrectly URL 
                # that end with the symbol '('. Therefore, is safer to use the field 'entity' to get the
                # entity embedding
                if 'dbr:' in w:
                    temp_list.append( vocab[ value['entity'].lower() ] )
                else:
                    temp_list.append(vocab[w])
            except KeyError:
                temp_list.append(vocab['<pad>'])
        try:
            temp_entity = vocab[ value['entity'].lower() ]
        except KeyError:
            temp_entity = vocab['<pad>']
        if(value['target'] == 'FALSE'):
            temp_target.append(0)
        else:
            temp_target.append(1)
        temp_rank.append(value['rank'])
        instance['source'] = temp_list
        instance['entity'] = temp_entity
        instance['target'] = temp_target
        instance['rank'] = temp_rank
        converted_dataset.append(instance)
    return converted_dataset

def get_word2idx_vocab(vocab):
    vocab2idx = dict()
    idx = 0
    for word in vocab:
        word = word.lower()
        vocab2idx[word] = idx
        idx += 1
    return vocab2idx

def build_weight_matrix(fast_text, target_vocab, embed_dim):
    matrix_len = len(target_vocab)
    weight_matrix = np.zeros((matrix_len, embed_dim))
    words_found = 0
    for i, word in enumerate(target_vocab):
        word = word.lower()
        try:
            weight_matrix[i] = fast_text[word]
            words_found += 1
        except KeyError:
            weight_matrix[i] = np.zeros(embed_dim)
    weight_tensor = torch.from_numpy(weight_matrix)
    return weight_tensor

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, optimizer, loss_function, clip, train_iter, phase):
    if phase == 'train':
        model = model.train()
    else: 
        model = model.eval()
    
    epoch_loss = 0
    
    for batch in train_iter:    
        X = Variable(batch[0].to(device)) # X = [batch_size, sentence_length] Ex: [20, 41]
        Y = Variable(batch[1].to(device)) # Y = [batch_size, label_length] Ex: [20, 1]
        rank = Variable(batch[2].to(device))
        Y = Y.squeeze() # Y = [batch_size]
        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            tag_scores = model(X, rank)
            loss = loss_function(tag_scores, Y)
        if phase == 'train':
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_iter)

def evaluate(model, loss_function, test_iter):
    model.eval()
    epoch_loss = 0
    class_correct = [0, 0]
    class_total = [0, 0]
    classes = [0, 1]
    auc_epoch = []
    with torch.no_grad():
        for i, batch in enumerate(test_iter):
            X = Variable(batch[0].to(device)) # X = [sentence_length, batch_size]
            Y = Variable(batch[1].to(device)) # Y = [batch_size, label_length]
            rank = Variable(batch[2].to(device))
            Y = Y.squeeze()
            tag_scores = model(X, rank) # tag_scores = [batch_size, label_length]
            if torch.cuda.is_available():
                auc = roc_auc_score(Y.cpu().numpy(), tag_scores.cpu().numpy())
            else:
                auc = roc_auc_score(Y.numpy(), tag_scores.numpy())
            auc_epoch.append(auc)
            loss = loss_function(tag_scores, Y)
            epoch_loss += loss.item()

            for i in range(tag_scores.shape[0]):
                if(tag_scores[i].item()>0.5):
                    pred = 1
                else:
                    pred = 0
                out = int(Y[i].item())
                if(pred == out):
                    class_correct[pred] += 1
                    class_total[pred] += 1
                else:
                    class_total[out] += 1
    for i in range(2):
        if class_total[i] > 0:
            print('Test Accuracy of {}: {} ({}/{})'.format(classes[i], (class_correct[i]*100)/class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of {}: N/A (no training example)'.format(classes[i]))
    print('Test Accuracy (Overall): {} ({}/{})\n'.format((np.sum(class_correct) * 100 / np.sum(class_total)), np.sum(class_correct), np.sum(class_total)))
    return epoch_loss / len(test_iter), class_correct, class_total, auc_epoch

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins*60))
    return elapsed_mins, elapsed_secs

if __name__ == '__main__':
    # Setting the arguments of the script
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--output_dim', type=int, default=None)
    parser.add_argument('--embedding', type=int,  default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--hidden', type=int, default=None)
    parser.add_argument('--layer', type=int, default=None)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--ws', type=int, default=None)
    parser.add_argument('--rank', default=None) # rank is the popularity

    args = parser.parse_args()

    # Define the path for the data to be loaded, like training/test set and the pre-trained embedding
    data_path = '/example/data/path/'
    dataset_name = args.dataset

    with open('{}train_{}.json'.format(data_path, dataset_name), 'r') as training_file:
        training_set = json.load(training_file)
    with open('{}test_{}.json'.format(data_path, dataset_name), 'r') as test_file:
        test_set = json.load(test_file)

    vector = bcolz.open('{}dbpedia{}.dat'.format(data_path, args.embedding))[:]
    vocab = pickle.load(open('{}dbpedia{}_words.pkl'.format(data_path, args.embedding), 'rb'))
    word2idx = pickle.load(open('{}dbpedia{}_idx.pkl'.format(data_path, args.embedding), 'rb'))
    
    fast_text = {w.lower(): vector[word2idx[w]] for w in vocab}
    weight_matrix = build_weight_matrix(fast_text, vocab, args.embedding)
    vocab2idx = get_word2idx_vocab(vocab)

    # Define the constant to build the Neural Network
    BATCH_SIZE = args.batch
    OUTPUT_DIM = args.output_dim
    EMBED_DIM = args.embedding
    HID_DIM = args.hidden
    N_LAYERS = args.layer
    DROPOUT = args.dropout
    N_EPOCH = args.epoch
    WS = args.ws
    RANK = args.rank

    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the path for the results and log files
    logging.basicConfig(filename='info_{}_{}_{}.log'.format(HID_DIM, N_LAYERS, dataset_name), level=logging.INFO, \
        format='%(asctime)s - %(message)s', datefmt='%y-%m-%d %H:%M:%S')

    result_file = open('result_{}_{}_{}_{}_{}.txt'.format(EMBED_DIM, N_EPOCH, HID_DIM, N_LAYERS, dataset_name), 'a')
    auc_file = open('auc_{}_{}_{}_{}_{}.txt'.format(EMBED_DIM, N_EPOCH, HID_DIM, N_LAYERS, dataset_name), 'a')
    loss_file = open('loss_{}_{}_{}_{}_{}.txt'.format(EMBED_DIM, N_EPOCH, HID_DIM, N_LAYERS, dataset_name), 'a')

    # Translate the training set to indexes
    if WS is not None:
        training_set = trim_dataset(training_set, WS)
    converted_training_set = convert_dataset(training_set, vocab2idx)
    train_dataset = DatasetOya(converted_training_set)

    valid_size = int(len(train_dataset)*0.2)
    train_size = len(train_dataset) - valid_size

    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    #Translate the test set to indexes
    if args.ws is not None:
        test_set = trim_dataset(test_set, args.ws)
    converted_test_set = convert_dataset(test_set, vocab2idx)
    test_dataset = DatasetOya(converted_test_set)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    logging.info("Parameters values:")
    logging.info("EPOCH: {}".format(N_EPOCH))
    logging.info("BATCH_SIZE: {}".format(BATCH_SIZE))
    logging.info("OUTPUT_DIM: {}".format(OUTPUT_DIM))
    logging.info("EMBEDDING DIMENSION: {}".format(EMBED_DIM))
    logging.info("HIDDEN DIMENSION: {}".format(HID_DIM))
    logging.info("NUMBER OF LAYERS: {}".format(N_LAYERS))
    logging.info("WEIGHT MATRIX: {}".format(weight_matrix.shape))
    logging.info("PADDING IDX: {}".format(vocab2idx['<pad>']))
    if WS is not None:
        logging.info('WINDOW CONTEXT: {}'.format(WS))
    if RANK is not None:
        logging.info('RANK: TRUE')
    logging.info('DEVICE: {}'.format(device))

    print('Hidden dimension: {}'.format(HID_DIM))
    print('Number of layers: {}'.format(N_LAYERS))

    # Initiate the model
    lstm = wordLSTM(EMBED_DIM, HID_DIM, N_LAYERS, DROPOUT, vocab2idx['<pad>'], BATCH_SIZE, weight_matrix).to(device)
    ffnn = FFNN(HID_DIM).to(device)
    optic = Optic(lstm, ffnn, device)

    # Initiate the model's weights
    optic.apply(init_weights)
    logging.info("The model has {} trainable parameters".format(count_parameters(optic)))

    # Set the loss function and the optimizer algorithm
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(optic.parameters(), lr=0.001)

    # Initializing loss vectors
    train_loss = []
    val_loss = []
    test_loss = []
    auc_score = []

    # Start training
    for epoch in range(N_EPOCH):
        start_time = time.time()
        # Although several tutorials state that is necessary to initialize the hidden states for each epoch,
        # The pytorch documentation shows that the hidden states are set to 0 automatically if not they are not provided
        #model.hidden = model.init_hidden()
        loss = train(optic, optimizer, criterion, 1, train_loader, 'train')
        train_loss.append(loss)
        loss = train(optic, optimizer, criterion, 1, valid_loader, 'valid')
        val_loss.append(loss)
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        logging.info('Epoch: {} | Time: {}m {}s'.format(epoch+1, epoch_mins, epoch_secs))
        logging.info('\tTrain Loss: {} | Train PPL: {}'.format(train_loss[-1], math.exp(train_loss[-1])))
        logging.info('\tValid Loss: {} | Valid PPL: {}'.format(val_loss[-1], math.exp(val_loss[-1])))

    loss_file.write('{}\t{}\n'.format(train_loss[-1], val_loss[-1]))
    loss, class_correct, class_total, auc_epoch = evaluate(optic, criterion, test_loader)
    auc_score = auc_score + auc_epoch
    test_loss.append(loss)
    logging.info('| Test Loss: {} | Test PPL: {} |'.format(test_loss[-1], math.exp(test_loss[-1])))

    # Write the result for each classes (0 = False, 1 = True)
    results = []
    for k in range(2):
        if class_total[k] > 0:
            results.append( (class_correct[k]*100)/class_total[k] )
        else:
            results.append(0)
    results.append( (np.sum(class_correct) * 100 / np.sum(class_total)) )
    result_file.write('{}\t{}\n'.format(HID_DIM, N_LAYERS))
    result_file.write('{}\t{}\t{}\n'.format(results[0], results[1], results[2]))
    auc_file.write('{}'.format(auc_score[-1]))
    
    # Close all open files
    result_file.close()
    auc_file.close()
    loss_file.close()

    #Generate graph
    plt = generate_loss_graph(train_loss, val_loss, 'Loss in {}/{}'.format(HID_DIM, N_LAYERS), 'Epoch', 'Loss')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.04), shadow=True, ncol=2)
    save_plot(plt, './loss_{}_{}.svg'.format(HID_DIM, N_LAYERS))
    save_plot(plt, './loss_{}_{}.png'.format(HID_DIM, N_LAYERS), 'png')
    plt.close()

    plt = generate_auc_graph(auc_score, 'AUC score in {}/{}'.format(HID_DIM, N_LAYERS), 'Epoch', 'AUC')
    save_plot(plt, './auc_{}_{}.svg'.format(HID_DIM, N_LAYERS))
    save_plot(plt, './auc_{}_{}.png'.format(HID_DIM, N_LAYERS), 'png')

    # Save the model 
    torch.save(optic.state_dict(), '{}optic_model.pth'.format(data_path))