# -*- coding: utf-8 -*-


from util import *

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from model import NN
import numpy as np

# Dataset class for pytorch
class DatasetOya(Dataset):
    def __init__(self, file):
        self.data = file

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = torch.tensor(self.data[idx]['source'], dtype=torch.long)
        rank = torch.tensor(self.data[idx]['rank'], dtype=torch.float)
        return text, rank

# Build a dictionary that given a word return their index in the embedding matrix
def get_word2idx_vocab(vocab):
    vocab2idx = dict()
    idx = 0
    for word in vocab:
        # To assure that the word is lowercase
        word = word.lower()
        vocab2idx[word] = idx
        idx += 1
    return vocab2idx

def trim_sequence(text, ws):
    sequence = text.split()
    trimmed_value = []
    left_ws = ws
    right_ws = 1
    entity = [w for w in sequence if w.startswith('dbr:')]
    entity_idx = sequence.index(entity[0])
    # Add left context
    for i in range(ws):
        idx = entity_idx - left_ws
        if idx >= 0:
            trimmed_value.append(sequence[idx])
        left_ws = left_ws - 1
    # Add entity
    trimmed_value.append(sequence[entity_idx])
    # Add right context
    for i in range(ws):
        idx = entity_idx + right_ws
        if idx < len(sequence):
            trimmed_value.append(sequence[idx])
        right_ws = right_ws + 1
    sequence = trimmed_value
    return ' '.join(sequence)

# Transform a sentence in a sequence of index 
# To ensure consistency, we transform all words of the sentence to lowercase
# Our vocab also only have words in lowercase
def prepare_sequence(instance, vocab):
    X = []
    for w in instance.split():
        w = w.lower()
        try:
            X.append(vocab[w])
        except KeyError:
            X.append(vocab['<pad>'])
    return X

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

def disambiguate_mentions(tweet, threshold, model, device, vocab2idx, ws, ex_att, verbose):
    if verbose == 'yes':
        print('\n..:: Preparing Sentences for Disambiguation ::..')
    entities = []
    BATCH_SIZE = 1

    # Convert the tweets with theirs candidates in instance for the Neural Network model
    # This list will be a list of list with the instances for each mention
    instances_mentions = []
    for i, mention in enumerate(tweet.mentions):
        instances = []
        # If the mention does not have any candidate, already assign None to it
        if tweet.candidates[i] is None:
            instances_mentions.append(None)
            if verbose == 'yes':
                print('Mention {}: None'.format(i+1))
        else:
            for candidate in tweet.candidates[i]:
                # Replace the mention with the candidate
                instance = dict()
                instance['source'] = tweet.procText.replace(mention[0], " {} ".format(candidate[0]))
                instance['rank'] = candidate[1]
                if verbose == 'yes':
                    print('Mention {}: {}'.format(i+1, instance['source']))
                if (ws is not None) and ('dbr:' in instance['source']):
                    instance['source'] = trim_sequence(instance['source'], ws)
                instance['source'] = prepare_sequence(instance['source'], vocab2idx)
                if verbose == 'yes':
                    print('\t{}'.format(instance['source']))
                instances.append(instance)
            instances_mentions.append(instances)
    model.eval()
    if verbose == 'yes':
        print('\n..:: Disambiguation Scores ..::')
    for i, instances in enumerate(instances_mentions):
        # If there no instances, i.e., the mention got no candidate, assign NIL to it
        if instances is None:
            entities.append('NIL')
            if verbose == 'yes':
                print('Mention {}: NIL'.format(i+1))
        else:
            BATCH_SIZE = len(instances)
            test_set = DatasetOya(instances)
            test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
            with torch.no_grad():
                for batch in test_loader:
                    X = Variable(batch[0].to(device))
                    rank = Variable(batch[1].to(device))
                    rank = rank[:, None]
                    if ex_att == 1:
                        tag_score = model(X, rank)
                        torch_scores = torch.sigmoid(tag_score)
                    else:
                        torch_scores = torch.sigmoid(model(X))
            scores = torch_scores.tolist()
            if type(scores) is float:
                scores = []
                scores.append(torch_scores.tolist())
            if verbose == 'yes':
                print('Mention {}:'.format(i+1))
                for j, score in enumerate(scores):
                    print('\tCandidate: {} - {}'.format(tweet.candidates[i][j][0], score))
            # If there only one instance and the score is higher than the threshold, assign the candidate to the mention
            if len(scores) == 1 and scores[0] > threshold:
                entities.append(tweet.candidates[i][0][0])
                if verbose == 'yes':
                    print('\tDisambiguation: {}'.format(tweet.candidates[i][0][0]))
            elif len(scores) == 1 and scores[0] < threshold:
                entities.append('NIL')
                if verbose == 'yes':
                    print('\tDisambiguation: NIL')
            # If all the scores are the same, it means that the model could not distinguish the correct candidate for the mention
            # Therefore, we conclude that any candidate describe correctly the mention and assign NIL to it
            elif scores.count(scores[0]) == len(scores):
                entities.append('NIL')
                if verbose == 'yes':
                    print('\tDisambiguation: NIL')
            else:
                # Get the index of the highest score
                index_max = max(range(len(scores)), key=scores.__getitem__)
                if max(scores) < threshold:
                    entities.append('NIL')
                    if verbose == 'yes':
                        print('\tDisambiguation: NIL')
                else:
                    entities.append(tweet.candidates[i][index_max][0])
                    if verbose == 'yes':
                        print('\tDisambiguation: {}'.format(tweet.candidates[i][index_max][0]))
    return entities