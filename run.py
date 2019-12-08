# -*- coding: utf-8 -*-


import argparse
import pickle
import time
import configparser
import sys
from rdflib.graph import Graph
from pynif import NIFCollection
import re
from flask import (
    Flask, request, render_template, g, url_for, redirect, session, flash
)
import bcolz
import numpy as np

import os
import fnmatch

from datasetGeneration import datasetGeneration as DG
from recognizer import *
from selector import *
from disambiguator import *
from tweet import Tweet
from util import *
from model import *
from historicalContext import *

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from model import (NN1, NN2)

@app.route('/d2kb', methods=['POST', 'GET'])
def d2kb():
    data = request.data
    data = data.rstrip()
    data = data.lstrip()
    nif_post = NIFCollection.loads(data.decode('utf-8'), format='turtle')
    mentions = []
    for context in nif_post.contexts:
        tweet = Tweet()
        tweet.mentions = []
        tweet.idTweet = context.uri
        tweet.text = context.mention
        try:
            for phrase in context.phrases:
                single_mention = (phrase.mention, phrase.beginIndex, phrase.endIndex)
                mentions.append(single_mention)
        except:
            print('no mentions')
        if len(mentions) > 0:
            if VERBOSE == 'yes':
                print('\n\n:::: PREPROCESSING ::::\n\n')
            start = time.time()
            tweet = preprocessing_d2kb(tweet, mentions, VERBOSE)
            end = time.time()
            if VERBOSE == 'yes':
                print('Running time: {}'.format(end-start))
            if VERBOSE == 'yes':
                print('\n\n:::: ENTITY SELECTION ::::\n\n')
            start = time.time()
            tweet.candidates = select_candidates(tweet, vocab2idx, TYPE, MAX, BOOST, VERBOSE)
            end = time.time()
            if VERBOSE == 'yes':
                print('Running time: {}'.format(end-start))
            if VERBOSE == 'yes':
                print('\n\n:::: DISAMBIGUATION ::::\n\n')
            start = time.time()
            tweet.entities = disambiguate_mentions(tweet, THRESHOLD, model, device, vocab2idx, WS, EXTRA, VERBOSE)
            end = time.time()
            if VERBOSE == 'yes':
                print('Running time: {}'.format(end-start))
        collection_name = "http://optic.ufsc.br/"
        nif = annotation2nif(collection_name, tweet)
    return nif

if __name__ == "__main__":    
    # ******************************** #
    #Initialize variables
    # General
    VERBOSE = None
    MODE = None
    DATA_PATH = None
    INPUT_PATH = None
    
    # Neural Network
    EMBEDDING_DIM = None
    HIDDEN_DIM = None
    N_LAYERS = None
    DROPOUT = None
    BATCH_SIZE = None

    # Selection
    TYPE = None
    MAX = None
    BOOST = None

    # Disambiguation
    THRESHOLD = None
    WS = None
    EXTRA = None
    
    # ******************************** #
    # Load config file
    cfg_local = configparser.ConfigParser()
    try:
        cfg_local.read_file(open('config_local.ini', 'r'))
        
        VERBOSE = cfg_local['GENERAL']['verbose']
        MODE = cfg_local['GENERAL']['mode']
        DATA_PATH = fix_paths(cfg_local['GENERAL']['data_path'])
        INPUT_PATH = fix_paths(cfg_local['GENERAL']['input_path'])

        EMBEDDING_DIM = int(cfg_local['NEURALNETWORK']['embedding_dim'])
        HIDDEN_DIM = int(cfg_local['NEURALNETWORK']['hidden_dim'])
        N_LAYERS = int(cfg_local['NEURALNETWORK']['n_layers'])
        DROPOUT = float(cfg_local['NEURALNETWORK']['dropout'])
        BATCH_SIZE = int(cfg_local['NEURALNETWORK']['batch_size'])

        TYPE = cfg_local['SELECTION']['type']
        MAX = int(cfg_local['SELECTION']['max'])
        BOOST = int(cfg_local['SELECTION']['boost'])

        THRESHOLD = float(cfg_local['DISAMBIGUATION']['threshold'])
        WS = int(cfg_local['DISAMBIGUATION']['ws'])
        EXTRA = int(cfg_local['DISAMBIGUATION']['extra'])
    except:
        print('*** No configuration file or incorrect one. A sample one will be created, please fill this! ***')
        build_config_file()
        print('\nPossible values for each key:\n\n')
        print('verbose - yes, no\nmode - a2kb, d2kb\nmax - 0 ~ 10000 (will depend of your elasticsearch configuration)\ndata_path - ./')
        print('embedding_dim - 1 ~ 200*\nhidden_dim - 1 ~ 200*\nn_layers - 1 ~ 10*\ndropout - 0 ~ 1\n batch_size - 1 ~ 200*')
        print('* Will depend of the values used during the training of the neural network')
        print('type - single, multi')
        print('threshold - 0 ~ 1')
        sys.exit(0)

    # ******************************** #
    # Load terminal arguments
    # Takes priority over the config file
    parser = argparse.ArgumentParser(
        description="Optic: Knowledge graph-augmented entity linking approach"
    )
    parser.add_argument('--mode', metavar='', help='Type of experiment', choices={'a2kb', 'd2kb'}, default=None)
    parser.add_argument('--data', metavar='', help='Path for the data files used by OPTIC', default=None)
    parser.add_argument('--input', metavar='', help='Path for the files in NIF', default=None)
    parser.add_argument('--verbose', metavar='', help='Print process informations', choices={'yes', 'no'}, default=None)

    parser.add_argument('--embed', metavar='', help='Dimension for the embedding layer', type=int, default=None)
    parser.add_argument('--hidden', metavar='', help='Number of cells for hidden layer(s)', type=int, default=None)
    parser.add_argument('--layer', metavar='', help='Number of hidden layers', type=int, default=None)
    parser.add_argument('--dropout', metavar='', help='Dropout value', type=float, default=None)
    parser.add_argument('--batch', metavar='', help='Size of batch', type=int, default= None)

    parser.add_argument('--type', metavar='', help='Type of elasticsearch query', choices={'single', 'multi'}, default=None)
    parser.add_argument('--max', metavar='', help='Max of documents returned by elasticsearch queries', type=int, default=None)
    parser.add_argument('--boost', metavar='', help='Boost for exact match in multi-match queries', type=int, default=None)

    parser.add_argument('--threshold', metavar='', help='Threshold similarity for the candidate selection step',
                        type=float, default=None)
    parser.add_argument('--ws', metavar='', help='Size of the window context', type=int, default=None)
    parser.add_argument('--extra', metavar='', help='Flag for extra attributes in the NN model (0 = None, 1 = popularity)', choices={0, 1}, type=int, default=None)

    args = parser.parse_args()

    if args.mode is not None:
        MODE = args.mode
    if args.data is not None:
        DATA_PATH = fix_paths(args.data)
    if args.input is not None:
        INPUT_PATH = fix_paths(args.input)
    if args.verbose is not None:
        VERBOSE = args.verbose

    if args.embed is not None:
        EMBEDDING_DIM = args.embed
    if args.hidden is not None:
        HIDDEN_DIM = args.hidden
    if args.layer is not None:
        N_LAYERS = args.layer
    if args.dropout is not None:
        DROPOUT = args.dropout
    if args.batch is not None:
        BATCH_SIZE = args.batch
    
    if args.type is not None:
        TYPE = args.type
    if args.max is not None:
        MAX = args.max
    if args.boost is not None:
        BOOST = args.boost

    if args.threshold is not None:
        THRESHOLD = args.threshold
    if args.ws is not None:
        WS = args.ws
    if args.extra is not None:
        EXTRA = args.extra
    
    # ******************************** #
    # Load neural network model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vector = bcolz.open('{}dbpedia{}.dat'.format(DATA_PATH, EMBEDDING_DIM))[:]
    words = pickle.load(open('{}dbpedia{}_words.pkl'.format(DATA_PATH, EMBEDDING_DIM), 'rb'))
    word2idx = pickle.load(open('{}dbpedia{}_idx.pkl'.format(DATA_PATH, EMBEDDING_DIM), 'rb'))
    fast_text = {w.lower(): vector[word2idx[w]] for w in words}

    weight_matrix = build_weight_matrix(fast_text, words, EMBEDDING_DIM)
    vocab2idx = get_word2idx_vocab(words)

    lstm = NN.wordLSTM(EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, vocab2idx['<pad>'], BATCH_SIZE, weight_matrix).to(device)
    ffnn = NN.FFNN(HIDDEN_DIM).to(device)
    model = NN.Optic(lstm, ffnn, device)

    if device == 'cpu':
        model.load_state_dict(torch.load('{}optic_model.pth'.format(DATA_PATH), map_location='cpu'))
    else:
        model.load_state_dict(torch.load('{}optic_model.pth'.format(DATA_PATH)))
    
    # ******************************** #
    # Start OPTIC
    count = 0

    # Read directory with tweets to be annotated
    inputs = set()
    for nif_temp in os.listdir(INPUT_PATH):
        # Initially, we works only with RDF turtle standard 
        if(fnmatch.fnmatch(nif_temp, '*.ttl')):
            inputs.add(nif_temp)

    for nif_input in inputs:
        nif_file = ''
        with open(INPUT_PATH + nif_input, 'r') as f:
            nif_file = f.read()
        nif_post = NIFCollection.loads(nif_file, format='turtle')
        for context in nif_post.contexts:
            tweet = Tweet()
            tweet.idTweet = context.uri
            tweet.text = context.mention
            tweet.mentions = []

            # A2KB Mode
            # TODO
            if MODE == 'a2kb':
                continue
            
            # D2KB Mode
            else:
                mentions = []
                try:
                    # Get all the mentions present in the tweet
                    for phrase in context.phrases:
                        single_mention = (phrase.mention, phrase.beginIndex, phrase.endIndex)
                        mentions.append(single_mention)
                # If there is no mention in tweet, return the original tweet
                except:
                    continue
                if len(mentions) > 0:
                    tweet = preprocessing_d2kb(tweet, mentions, VERBOSE)
                    tweet.candidates = select_candidates(tweet, vocab2idx, TYPE, MAX, BOOST, VERBOSE)
                    tweet.entities = disambiguate_mentions(tweet, THRESHOLD, model, device, vocab2idx, WS, EXTRA, VERBOSE)
            # Create tweet semantically annotated, as nif, when there are mentions
            # If not, just return the tweet as nif
            collection_name = "http://optic.ufsc.br/"
            nif = annotation2nif(collection_name, tweet)
            with open('{}output/{}.ttl'.format(DATA_PATH, count), 'w') as output_file:
                output_file.write(nif)
                count += 1
