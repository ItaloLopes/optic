# -*- coding: utf-8 -*-


import json
from tweet import Tweet
from collections import defaultdict
import pickle
from pynif import NIFCollection

def json_to_tweets(json_input):
    tweet = Tweet()
    with open(json_input, 'r') as jsonFile:
        data = json.load(jsonFile)
        for typeID, tweetID in data['id'].items():
            tweet.idTweet = int(tweetID)
            tweet.text = data['text']
    return tweet

def annotation2nif(collection_name, tweet):
    collection = NIFCollection(uri=collection_name)
    context_name = collection_name + str(tweet.idTweet)
    context = collection.add_context(uri=context_name, mention=tweet.text)
    if len(tweet.mentions) > 0:
        for i, mention in enumerate(tweet.mentions):
            if tweet.entities[i] != 'NIL':
                entity = tweet.entities[i].replace('dbr:', 'http://dbpedia.org/resource/')
            else:
                entity = 'http://optic.ufsc.br/resource/NIL/'
            context.add_phrase(
                beginIndex=int(mention[2]),
                endIndex=int(mention[3]),
                annotator='http://optic.ufsc.br',
                taIdentRef=entity
            )
    nif = collection.dumps(format='turtle')
    return nif

def nif_2_annotations(nif_collection):
    annotations = defaultdict(list)
    temp_annotations = defaultdict(list)
    keys = []

    parsed_collection = NIFCollection.loads(nif_collection, format='turtle')
    for context in parsed_collection.contexts:
        for phrase in context.phrases:
            id_annotation = phrase.context.rsplit('/', 1)[-1]
            entity = phrase.taIdentRef
            keys.append(int(id_annotation))
            temp_annotations[int(id_annotation)].append(entity)
    keys.sort()
    for key in keys:
        annotations[key] = temp_annotations[key]
    return annotations 

def text_to_tweet(text_input):
    tweet = Tweet()
    tweet.text = text_input
    return tweet

def print_tweet_status(tweet):
    print('Original text:\n{}'.format(tweet.text))
    print('..............')
    print('Pre-processed text:\n{}'.format(tweet.procText))
    print('..............')
    print('Original tags:\n{}'.format(tweet.tags))
    print('..............')
    print('Pre-processed tags:\n{}'.format(tweet.procTags))
    print('..............')
    print('Mentions:\n{}'.format(tweet.mentions))
    print('..............')
    print('Geocoordinates:\n{}'.format(tweet.geoCoord))
    print('..............')
    print('Candidates:\n{}'.format(tweet.candidates))
    print('..............')

def build_config_file():
    cfg_string = ''
    cfg_string += '[GENERAL]\nverbose= \nmode= \nmax= \ndata_path= \n'
    cfg_string += '\n[NEURALNETWORK]\nembedding_dim= \nhidden_dim= \nn_layers= \ndropout= \nbatch_size= \n'
    cfg_string += '\n[SELECTION]\ntype= \n'
    cfg_string += '\n[DISAMBIGUATION]\nthreshold= '
    
    try:
        with open('config_sample.ini', 'w') as f:
            f.write(cfg_string)
    except:
        print(' *** Failed to create the sample config file. Please verify the folder permissions!')

def fix_paths(path):
    if path[len(path) - 1] != '/':
            path += '/'
    return path
