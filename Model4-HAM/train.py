import time
import os
import shutil
import logging
import argparse
import random
import glob
import warnings
warnings.filterwarnings('always')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, classification_report
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)
import matplotlib.pyplot as plt

from models.harnn import HARNN
from utils.data_loader import TrainDataset
from utils import data_helper as dh

        
def save_model(model, is_best, filename):
    torch.save(model, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')
        
def save_loss_curve(args, train_losses, val_losses):
    plt.plot(np.arange(1, args.epochs+1), train_losses, label = "Train_loss")
    plt.plot(np.arange(1, args.epochs+1), val_losses, label = "Val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig(args.save_loss_curve)
    logging.info("Loss curve saved!")

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        self.MSELoss = nn.MSELoss()

    def forward(self, first_logits, second_logits, third_logits, global_logits,
                first_scores, second_scores, input_y_first, input_y_second, input_y_third,
                input_y):
        # Local Loss
        losses_1 = self.BCEWithLogitsLoss(first_logits, input_y_first.float())
        losses_2 = self.BCEWithLogitsLoss(second_logits, input_y_second.float())
        losses_3 = self.BCEWithLogitsLoss(third_logits, input_y_third.float())
        local_losses = 0.2*losses_1 + 0.3*losses_2 + 0.5*losses_3

        # Global Loss
        global_losses = self.BCEWithLogitsLoss(global_logits, input_y.float())

        # Hierarchical violation Loss
        return local_losses + global_losses
        
        
def train(args, net, criterion, optimizer, scheduler, train_loader, val_loader):
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(args.device)
    
    trained_model_dir = os.path.join(os.path.curdir, "result", "trained_model")
    for f in os.listdir(trained_model_dir):
        os.remove(os.path.join(trained_model_dir, f))

    logging.info("Training...")
    # writer = SummaryWriter('summary')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        train_loss = 0.0
        train_cnt = 0
        
        for train_iter, (x_train, y_train, y_train_0, y_train_1, y_train_2) in enumerate(train_loader):
            x_train, y_train, y_train_0, y_train_1, y_train_2 = \
                [i.to(args.device) for i in [x_train, y_train, y_train_0, y_train_1, y_train_2]]

            _, outputs = net(x_train)
            loss = criterion(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5],
                             y_train_0, y_train_1, y_train_2, y_train)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            train_loss += loss.item()
            train_cnt += x_train.size()[0]
            
            if train_iter % args.print_every == 0:
                logging.info('[%d, %5d] loss: %.3f' % (epoch + 1, train_cnt + 1, train_loss / train_cnt))
        
        eval_prc = 0
        if epoch % args.evaluate_every == 0:
            val_loss, eval_auc, eval_prc = eval(args, val_loader, net, criterion, is_save_cls_report = True)
            
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        is_best = eval_prc > args.best_prc
        args.best_prc = max(eval_prc, args.best_prc)
        if epoch % args.checkpoint_every == 0:
            timestamp = str(int(time.time()))
            save_model(net, is_best, filename=os.path.join(os.path.curdir, "result", "trained_model", "epoch%d.%s.pth" % (epoch, timestamp)))
    
    save_loss_curve(args, train_losses, val_losses)
    logging.info('Finished Training.')
    
    
def eval(args, val_loader, net, criterion, is_save_cls_report = False):
    val_loss = 0.0
    val_cnt = 0
    eval_pre_tk = [0.0 for _ in range(args.top_num)]
    eval_rec_tk = [0.0 for _ in range(args.top_num)]
    eval_F_tk = [0.0 for _ in range(args.top_num)]
    true_onehot_labels = []
    predicted_onehot_scores = []
    predicted_onehot_labels_ts = []
    predicted_onehot_labels_tk = [[] for _ in range(args.top_num)]
    for x_val, y_val, y_val_0, y_val_1, y_val_2 in val_loader:
        x_val, y_val, y_val_0, y_val_1, y_val_2 = \
            [i.to(args.device) for i in [x_val, y_val, y_val_0, y_val_1, y_val_2]]
        scores, outputs = net(x_val)
        scores = scores[0]
        loss = criterion(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5],
                         y_val_0, y_val_1, y_val_2, y_val)
        val_loss += loss.item()
        val_cnt += x_val.size()[0]
        # Prepare for calculating metrics
        for onehot_labels in y_val:
            true_onehot_labels.append(onehot_labels.tolist())
        for onehot_scores in scores:
            predicted_onehot_scores.append(onehot_scores.tolist())
        # Predict by threshold
        batch_predicted_onehot_labels_ts = \
            dh.get_onehot_label_threshold(scores=scores.cpu().detach().numpy(), threshold=args.threshold)
        for onehot_labels in batch_predicted_onehot_labels_ts:
            predicted_onehot_labels_ts.append(onehot_labels)
        # Predict by topK
        for num in range(args.top_num):
            batch_predicted_onehot_labels_tk = \
                dh.get_onehot_label_topk(scores=scores.cpu().detach().numpy(), top_num=num + 1)
            for onehot_labels in batch_predicted_onehot_labels_tk:
                predicted_onehot_labels_tk[num].append(onehot_labels)

    # Calculate Precision & Recall & F1
    eval_pre_ts = precision_score(y_true=np.array(true_onehot_labels),
                                  y_pred=np.array(predicted_onehot_labels_ts), average='micro')
    eval_rec_ts = recall_score(y_true=np.array(true_onehot_labels),
                               y_pred=np.array(predicted_onehot_labels_ts), average='micro', zero_division = 0)
    eval_F_ts = f1_score(y_true=np.array(true_onehot_labels),
                         y_pred=np.array(predicted_onehot_labels_ts), average='micro', zero_division = 0)
    # Calculate the average AUC
    eval_auc = roc_auc_score(y_true=np.array(true_onehot_labels),
                             y_score=np.array(predicted_onehot_scores), average='micro')
    # Calculate the average PR
    eval_prc = average_precision_score(y_true=np.array(true_onehot_labels),
                                       y_score=np.array(predicted_onehot_scores), average='micro')

    for num in range(args.top_num):
        eval_pre_tk[num] = precision_score(y_true=np.array(true_onehot_labels),
                                           y_pred=np.array(predicted_onehot_labels_tk[num]), average='micro')
        eval_rec_tk[num] = recall_score(y_true=np.array(true_onehot_labels),
                                        y_pred=np.array(predicted_onehot_labels_tk[num]), average='micro', zero_division = 0)
        eval_F_tk[num] = f1_score(y_true=np.array(true_onehot_labels),
                                  y_pred=np.array(predicted_onehot_labels_tk[num]), average='micro', zero_division = 0)
    logging.info("All Validation set: Loss {0:g} | AUC {1:g} | AUPRC {2:g}"
                .format(val_loss / val_cnt, eval_auc, eval_prc))
    logging.info("Predict by threshold: Precision {0:g}, Recall {1:g}, F {2:g}"
                .format(eval_pre_ts, eval_rec_ts, eval_F_ts))
    logging.info("Predict by topK:")
    for num in range(args.top_num):
        logging.info("Top{0}: Precision {1:g}, Recall {2:g}, F {3:g}"
                    .format(num + 1, eval_pre_tk[num], eval_rec_tk[num], eval_F_tk[num]))
                    
    if is_save_cls_report:
        target_names = ['not sexist', 'sexist', 
                            'none', '1. threats, plans to harm and incitement','2. derogation', '3. animosity', '4. prejudiced discussions',
                            'none',
                            '1.1 threats of harm',
                            '1.2 incitement and encouragement of harm',
                            '2.1 descriptive attacks', 
                            '2.2 aggressive and emotive attacks', 
							'2.3 dehumanising attacks & overt sexual objectification', 
							'3.1 casual use of gendered slurs, profanities, and insults', 
							'3.2 immutable gender differences and gender stereotypes', 
                            '3.3 backhanded gendered compliments', 
							'3.4 condescending explanations or unwelcome advice', 
							'4.1 supporting mistreatment of individual women', 
                            '4.2 supporting systemic discrimination against women as a group']
        #print(classification_report(y_true=np.array(true_onehot_labels), y_pred=np.array(predicted_onehot_labels_ts), target_names=target_names))
        clsf_report = pd.DataFrame(classification_report(y_true=np.array(true_onehot_labels), y_pred=np.array(predicted_onehot_labels_ts), target_names=target_names, output_dict=True)).transpose()
        clsf_report.to_csv(args.save_classification_report_filepath, index= True)
        #print("Saved classification report to : ", args.save_classification_report_filepath)
    
    return val_loss, eval_auc, eval_prc


def run(args):
    
    logging.info("Loading Data...")
    args.train_file_path, args.valid_file_path = dh.train_val_split(args.input_file_path, args.split_ration)
    
    train_dataset = TrainDataset(args, args.train_file_path)
    valid_dataset = TrainDataset(args, args.valid_file_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    
    vocab_size = dh.get_vocab_size(args.train_file_path, args.valid_file_path)
    
    logging.info("Init nn...")
    net = HARNN(num_classes_list=args.num_classes_layer, total_classes=args.total_classes, vocab_size=vocab_size,
                    embedding_size=args.embedding_size, lstm_hidden_size=args.lstm_hidden_size,
                    attention_unit_size=args.attention_unit_size,
                    fc_hidden_size=args.fc_hidden_size, beta=args.beta,
                    drop_prob=args.drop_prob)
                    
    criterion = Loss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg_lambda, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_loader)*args.epochs)
    
    if args.is_eval:
        print("Evaluationg model...")
        net = torch.load(args.best_model_file_path)
        print("Model loaded! : ", args.best_model_file_path)
        net.to(args.device)
        eval(args, val_loader, net, criterion, is_save_cls_report = True)
    else:
        train(args, net, criterion, optimizer, scheduler, train_loader, val_loader)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.deterministic = False

def main():
    #----------------- DO NOT CHANGE THESE PARAM -----------------------------
    logging.basicConfig(
    filename='./logs/model_log.txt',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser()
    #     args = parser.parse_args()
    args = parser.parse_args(args=[])
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.seed = 42
    set_seed(args.seed)

    args.num_classes_layer = [2, 5, 12]
    args.total_classes = 19
    
    args.input_file_path = os.path.join(os.path.curdir, "data", "train_all_tasks.csv")
    #args.input_file_path = os.path.join(os.path.curdir, "data", "downsample_train_all_tasks.csv")
    args.split_ration = 0.3
    
    args.best_model_file_path = os.path.join(os.path.curdir, "result", "model_best.pth")
    args.save_classification_report_filepath = os.path.join(os.path.curdir, "result", "classification_report.csv")
    args.save_loss_curve = os.path.join(os.path.curdir, "result", "loss_curve.png")
    #------------------------------------------------------------------------------
    
    
    # -------------------- YOU MAY EDIT THE BELOW PARAM ----------------------------
    args.is_eval = False # False for training, True for evaluation

    args.print_every = 1
    args.evaluate_every = 1
    args.checkpoint_every = 2

    args.embedding_size = 400
    args.seq_length = 128

    args.batch_size = 16
    args.epochs = 15
    args.max_grad_norm = 0.1
    args.drop_prob = 0.3
    args.l2_reg_lambda = 0
    args.learning_rate = 0.0001
    args.beta = 0.3

    args.lstm_hidden_size = 128
    args.fc_hidden_size = 64

    args.attention_unit_size = 64

    args.threshold = 0.5
    args.top_num = 3
    args.best_prc = 0
    # ---------------------------------------------------------------------------------


    run(args)

if __name__=='__main__':
    main()
