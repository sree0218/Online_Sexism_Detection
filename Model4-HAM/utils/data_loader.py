import json
from collections import Counter

import torch
import torch.optim
from torch.utils.data.dataset import Dataset

import numpy as np
from utils import data_helper
import pandas as pd

UNK_TOKEN = "<UNK>"

class InputFeature(object):
    def __init__(self,
                 _id,
                 features,
                 labels,
                 onehot_labels_tuple_list,
                 onehot_labels_list) -> None:
        self.id = _id
        self.features = features
        self.labels = labels
        self.onehot_labels_tuple_list = onehot_labels_tuple_list
        self.onehot_labels_list = onehot_labels_list
        
class_A_labels_to_idx = {'not sexist':0, 'sexist':1}
class_B_labels_to_idx = {'none':0, '1. threats, plans to harm and incitement':1,'2. derogation':2, '3. animosity':3, '4. prejudiced discussions':4}
class_C_labels_to_idx = {'none':0,
                            '1.1 threats of harm':1,
                            '1.2 incitement and encouragement of harm':2,
                            '2.1 descriptive attacks':3, 
                            '2.2 aggressive and emotive attacks':4, 
							'2.3 dehumanising attacks & overt sexual objectification':5, 
							'3.1 casual use of gendered slurs, profanities, and insults':6, 
							'3.2 immutable gender differences and gender stereotypes':7, 
                            '3.3 backhanded gendered compliments':8, 
							'3.4 condescending explanations or unwelcome advice':9, 
							'4.1 supporting mistreatment of individual women':10, 
                            '4.2 supporting systemic discrimination against women as a group':11}

def _pad_features(texts_ints, seq_length):
    features = np.zeros((1, seq_length), dtype=int)

    #features[0,-texts_ints.shape[1]:] = np.array(texts_ints)[:min(seq_length, len(text_ints))]
    features[0,-texts_ints.shape[1]:] = np.array(texts_ints)[:min(seq_length, texts_ints.shape[1])]

    # for i, row in enumerate(texts_ints):
    #     features[i, -len(row):] = np.array(row)[:seq_length]
    return features.reshape(-1)


def _create_onehot_labels(labels_index, num_labels):
    label = [0] * num_labels
    for item in labels_index:
        label[int(item)] = 1
    return label


def convert_examples_to_features(datapoint, args, vocab_to_int, is_test = False):
    text_data = datapoint[1]
    texts_ints = []
    for word in text_data.split():
        if word in vocab_to_int:
            texts_ints.append(vocab_to_int[word])
        else:
            texts_ints.append(vocab_to_int[UNK_TOKEN])
    texts_ints = np.array(texts_ints).reshape(1, -1)
    
    features = _pad_features(texts_ints,seq_length=args.seq_length)
    
    if is_test:
        return InputFeature(datapoint[0], features, None, None, None)
    else:
        onehot_labels_tuple_list = (_create_onehot_labels([class_A_labels_to_idx[datapoint[2]]], args.num_classes_layer[0]),
                                _create_onehot_labels([class_B_labels_to_idx[datapoint[3]]], args.num_classes_layer[1]),
                                _create_onehot_labels([class_C_labels_to_idx[datapoint[4]]], args.num_classes_layer[2]))
    
        onehot_labels_list = (_create_onehot_labels(datapoint[5].strip('\n').split(), args.total_classes))
        return InputFeature(datapoint[0], features, datapoint[5].strip('\n').split(), onehot_labels_tuple_list, onehot_labels_list )


class TrainDataset(Dataset):
    def __init__(self, args, file_path) -> None:
        self.examples = []
        data = []
        with open(file_path, encoding="utf8") as f:
            for line in f:
                line_splitted = line.split('\t') # rewire_id, text, label_sexist, label_category, label_vector
                line_splitted[1] = data_helper.clean_text(line_splitted[1])
                if len(line_splitted[1]) > 5:
                    data.append(line_splitted)
        
        vocab_to_int = data_helper.create_vocab(data)
        self.vocab_size = len(vocab_to_int)

        for datapoint in data:
            self.examples.append(convert_examples_to_features(datapoint,args, vocab_to_int, False))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return (torch.tensor(self.examples[index].features),
                torch.tensor(self.examples[index].onehot_labels_list),
                torch.tensor(self.examples[index].onehot_labels_tuple_list[0]),
                torch.tensor(self.examples[index].onehot_labels_tuple_list[1]),
                torch.tensor(self.examples[index].onehot_labels_tuple_list[2]))
                
                
class TestDataset(Dataset):
    def __init__(self, args, train_file_path, test_file_path) -> None:
        self.examples = []
        train_data = []
        test_data = []
        with open(train_file_path, encoding="utf8") as f:
            for line in f:
                line_splitted = line.split('\t') # rewire_id, text
                line_splitted[1] = data_helper.clean_text(line_splitted[1])
                if len(line_splitted[1]) > 5:
                    train_data.append(line_splitted)
        
        vocab_to_int = data_helper.create_vocab(train_data)
        self.vocab_size = len(vocab_to_int)
        
        with open(test_file_path, encoding="utf8") as f:
            for line in f:
                line_splitted = line.split('\t') # rewire_id, text
                line_splitted[1] = data_helper.clean_text(line_splitted[1])
                if len(line_splitted[1]) > 1:
                    test_data.append(line_splitted)

        for datapoint in test_data:
            self.examples.append(convert_examples_to_features(datapoint,args, vocab_to_int, True))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return (torch.tensor(self.examples[index].features))

# args={'file_path':'data/validation_sample.json', 'seq_length':200, 'num_classes_layer':[9, 128, 661, 8364], 'total_classes':9162}
# dataset = TextDataset(args, args['file_path'])
# print(dataset.__len__())