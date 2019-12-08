import numpy as np
import bcolz
import pickle

import torch
import torch.nn as nn

EMBEDDING_DIM = 200 # Example of dimension
data_path = '/example/data/path/'

def prepare_embedding():
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir='{}dbpedia{}.dat'.format(data_path, EMBEDDING_DIM), mode='w')
    miss_vocab = 0

    with open('{}dbpedia{}.vec'.format(data_path, EMBEDDING_DIM), 'r') as file:
        for line in file:
            line = line.rstrip()
            columns = line.split(" ")
            word = columns[0]
            word = word.lower()
            if(len(columns[1:]) == EMBEDDING_DIM):
                words.append(word)
                word2idx[word] = idx
                idx +=1
                vect = np.array(columns[1:]).astype(np.float)
                vectors.append(vect)
    #add padding to the vocab
    word = "<pad>"
    words.append(word)
    word2idx[word] = idx
    idx += 1
    vect = np.zeros(EMBEDDING_DIM).astype(np.float)
    print(vect)
    vectors.append(vect)
    size_vocab = len(words)
    vectors = bcolz.carray(vectors[1:].reshape((size_vocab, EMBEDDING_DIM)), rootdir='{}dbpedia{}.dat'.format(data_path, EMBEDDING_DIM), mode='w')
    vectors.flush()
    pickle.dump(words, open('{}dbpedia{}_words.pkl'.format(data_path, EMBEDDING_DIM), 'wb'))
    pickle.dump(word2idx, open('{}dbpedia{}_idx.pkl'.format(data_path, EMBEDDING_DIM), 'wb'))

    fast_text = {w: vectors[word2idx[w]] for w in words}
    return fast_text, size_vocab

def build_weight_matrix(fast_text, target_vocab):
    matrix_len = len(target_vocab)
    weight_matrix = np.zeros((matrix_len+1, EMBEDDING_DIM))
    words_found = 0
    for i, word in enumerate(target_vocab):
        try:
            weight_matrix[i] = fast_text[word]
            words_found += 1
        except KeyError:
            weight_matrix[i] = np.zeros(EMBEDDING_DIM)
    weight_tensor = torch.from_numpy(weight_matrix)
    return weight_tensor

def get_target_vocab(training_path):
    training_set = pickle.load(open(training_path, 'rb'))
    set_vocab = set()
    for instance in training_set:
        tokens_tweet = instance['source']
        for token in tokens_tweet:
            set_vocab.add(token)
    set_vocab.add("<PAD>")
    return list(set_vocab)

def create_emb_layer(weight_matrix, non_train=False):
    num_embeddings, embedding_dim = weight_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weight_matrix})
    if non_train:
        emb_layer.weight.requires_grad = False
    
    return emb_layer, num_embeddings, embedding_dim

fast_text, size_vocab = prepare_embedding()
print(size_vocab)
