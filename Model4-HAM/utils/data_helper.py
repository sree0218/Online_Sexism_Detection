import numpy as np
import pandas as pd
import heapq
import json
import re
import os
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

UNK_TOKEN = "<UNK>"

def get_unique_task_A_label():
    return ['not sexist', 'sexist']
    
def get_unique_task_B_label():
    return ['none', '2. derogation', '1. threats, plans to harm and incitement', '3. animosity', '4. prejudiced discussions']

def get_unique_task_C_label():
    return ['none',
            '2.3 dehumanising attacks & overt sexual objectification', 
            '2.1 descriptive attacks', 
            '1.2 incitement and encouragement of harm', 
            '3.1 casual use of gendered slurs, profanities, and insults', 
            '4.2 supporting systemic discrimination against women as a group', 
            '2.2 aggressive and emotive attacks', 
            '3.2 immutable gender differences and gender stereotypes', 
            '3.4 condescending explanations or unwelcome advice', 
            '3.3 backhanded gendered compliments', 
            '4.1 supporting mistreatment of individual women', 
            '1.1 threats of harm']
            
def get_TSV_filepath(filepath):
    head, tail = os.path.split(filepath)
    tsv_filename = ".".join(tail.split(".")[:-1]) + ".tsv"
    return os.path.join(head, tsv_filename)

def get_CSV_filepath(filepath, new_filename):
    head, tail = os.path.split(filepath)
    return os.path.join(head, new_filename)  

def plot_histogram(y):
    plt.figure(figsize=(15, 10), dpi=80)
    plt.hist(y, rwidth=0.7)
    plt.ylabel('Frequency')
    plt.xlabel('Labels')
    plt.xticks(rotation=90)
    #plt.yticks(range(0, 650, 50))
    plt.title('Histogram of a Labels')
    plt.show()
            
def train_val_split(filepath, split_ration = 0.3):
    df = pd.read_csv(filepath)
    X, y = df.index, df['label_vector']
    x_train_idx, x_val_idx, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify = y)
    #plot_histogram(y_train)
    #plot_histogram(y_val)
    
    val = df.loc[x_val_idx]
    train = df.drop(x_val_idx) 
    train_filepath = get_CSV_filepath(filepath, "train.csv")
    val_filepath = get_CSV_filepath(filepath, "val.csv")
    train.to_csv(train_filepath, index = False)
    val.to_csv(val_filepath, index = False)
    return csv_to_tsv(train_filepath), csv_to_tsv(val_filepath)


def csv_to_tsv(csv_filepath):
    df = pd.read_csv(csv_filepath)
    def all_labels(l1, l2, l3):
        unique_label_sexist = get_unique_task_A_label()
        unique_label_category = get_unique_task_B_label()
        unique_label_vector = get_unique_task_C_label()
        l1_ind = unique_label_sexist.index(l1)
        l2_ind = len(unique_label_sexist) + unique_label_category.index(l2)
        l3_ind = len(unique_label_sexist) + len(unique_label_category) + unique_label_vector.index(l3)
        return " ".join(list(map(str,[l1_ind, l2_ind, l3_ind])))
    df['labels'] = df.apply(lambda x: all_labels(x['label_sexist'], x['label_category'], x['label_vector']), axis=1)
    tsv_filename = get_TSV_filepath(csv_filepath)
    df.to_csv(tsv_filename, sep = "\t", index = False, header = False)
    if os.path.isfile(csv_filepath):
        os.remove(csv_filepath)
    return tsv_filename
    
def downsample_data(csv_filepath):
    data = pd.read_csv(csv_filepath)
    df = data[data.label_sexist == 'sexist']
    ndf = data[data.label_sexist == 'not sexist']
    add_df = ndf.sample(400)
    frames = [add_df,df]
    data = pd.concat(frames)
    data.to_csv(get_CSV_filepath(csv_filepath, "downsample_train_all_tasks.csv"), index = False)

def clean_text(text):
	text = re.sub('[URL]', '', text) # remove [URL]
	text = re.sub('[USER]', '', text) # remove [USER]
	text = re.sub(r'<.*?>', '', text)  # remove html tags
	text = text.lower() #lower case
	text = re.sub(r'http\S+', '', text) # remove http links
	text = re.sub(r'www\S+', '', text)  # remove www website
	text = re.sub(r'[^\w\s]', '', text) # remove special characters like !,@,#,$,%
	text = re.sub('\s+', ' ', text) # replace multiple space by single space
	return text

def create_vocab(data):
    texts_split = [datapoint[1] for datapoint in data]
    all_text =' '.join([texts for texts in texts_split])

    words = all_text.split()
    counts = Counter(words)
    counts[UNK_TOKEN] = len(counts)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    return vocab_to_int

def get_vocab_size(train_file_path, valid_file_path):
    data = []
    with open(train_file_path, encoding="utf8") as f:
        for line in f:
            line_splitted = line.split('\t') # rewire_id, text, label_sexist, label_category, label_vector
            line_splitted[1] = clean_text(line_splitted[1])
            data.append(line_splitted)
    if valid_file_path:
        with open(valid_file_path, encoding="utf8") as f:
            for line in f:
                line_splitted = line.split('\t') # rewire_id, text, label_sexist, label_category, label_vector
                line_splitted[1] = clean_text(line_splitted[1])
                data.append(line_splitted)

    vocab_to_int = create_vocab(data)
    vocab_size = len(vocab_to_int)
    return vocab_size

def get_onehot_label_threshold(scores, threshold=0.5):
    """
    Get the predicted onehot labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.
    Args:
        scores: The all classes predicted scores provided by network
        threshold: The threshold (default: 0.5)
    Returns:
        predicted_onehot_labels: The predicted labels (onehot)
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        onehot_labels_list = [0] * len(score)
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                onehot_labels_list[index] = 1
                count += 1
        if count == 0:
            max_score_index = score.index(max(score))
            onehot_labels_list[max_score_index] = 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels


def get_onehot_label_topk(scores, top_num=1):
    """
    Get the predicted onehot labels based on the topK number.
    Args:
        scores: The all classes predicted scores provided by network
        top_num: The max topK number (default: 5)
    Returns:
        predicted_onehot_labels: The predicted labels (onehot)
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        onehot_labels_list = [0] * len(score)
        max_num_index_list = list(map(score.index, heapq.nlargest(top_num, score)))
        for i in max_num_index_list:
            onehot_labels_list[i] = 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels


def get_label_threshold(scores, threshold=0.5):
    """
    Get the predicted labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.
    Args:
        scores: The all classes predicted scores provided by network
        threshold: The threshold (default: 0.5)
    Returns:
        predicted_labels: The predicted labels
        predicted_scores: The predicted scores
    """
    predicted_labels = []
    predicted_scores = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        index_list = []
        score_list = []
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                index_list.append(index)
                score_list.append(predict_score)
                count += 1
        if count == 0:
            index_list.append(score.index(max(score)))
            score_list.append(max(score))
        predicted_labels.append(index_list)
        predicted_scores.append(score_list)
    return predicted_labels, predicted_scores


def get_label_topk(scores, top_num=1):
    """
    Get the predicted labels based on the topK number.
    Args:
        scores: The all classes predicted scores provided by network
        top_num: The max topK number (default: 5)
    Returns:
        The predicted labels
    """
    predicted_labels = []
    predicted_scores = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        score_list = []
        index_list = np.argsort(score)[-top_num:]
        index_list = index_list[::-1]
        for index in index_list:
            score_list.append(score[index])
        predicted_labels.append(np.ndarray.tolist(index_list))
        predicted_scores.append(score_list)
    return predicted_labels, predicted_scores