"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl

from train.metrics import accuracy_SBM as accuracy
from train.metrics import accuracy_smoothing


def smooth_train_epoch(model, optimizer, device, data_loader, epoch, prev_smoothed_labels, delta=1.0, onehot=False):
    if epoch == 0:
        # test code to debug
        # test_data = []
        # for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
        #     test_data.append(batch_labels.to(torch.float))
        model.train()
        epoch_loss = 0
        epoch_train_acc = 0
        smoothed_labels = []
        original_labels = []
        predicts = []
        for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_labels = batch_labels.to(device)
            batch_snorm_n = batch_snorm_n.to(device)  # num x 1
            batch_scores, smoothed_label = model.forward(g=batch_graphs, h=batch_x, e=batch_e, label=batch_labels,
                                                         delta=delta, snorm_e=batch_snorm_e, snorm_n=batch_snorm_n,
                                                         onehot=onehot)
            original_labels.append(original_labels)
            smoothed_labels.append(smoothed_label)
            predicts.append(batch_scores)
            loss = model.loss(batch_scores, smoothed_label, onehot=onehot)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            epoch_train_acc += accuracy_smoothing(batch_scores, smoothed_label)
        epoch_loss /= (iter + 1)
        epoch_train_acc /= (iter + 1)
        with open(f'train_smoothed_labels_{epoch}.csv', 'w') as f:
            f.write("GT,smoothed Label,predict\n")
            for origin, smooth, predict in zip(original_labels, smoothed_labels, predicts):
                f.write(str(origin) + "," + str(smooth) + "," + str(predict) + "\n")
        return epoch_loss, epoch_train_acc, optimizer, smoothed_labels
        # test code to debug
        # return 0, 0, optimizer, test_data # for test

    # epoch > 0
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    original_labels = []
    smoothed_labels = []
    predicts = []

    for iter, (batch_graphs, batch_labels , batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        original_labels.append(batch_labels)
        batch_labels = prev_smoothed_labels[iter].to(device)
        batch_snorm_n = batch_snorm_n.to(device)  # num x 1
        batch_scores, smoothed_label = model.forward(g=batch_graphs, h=batch_x, e=batch_e, label=batch_labels,
                                                     delta=delta, snorm_e=batch_snorm_e, snorm_n=batch_snorm_n,
                                                     onehot=onehot)
        smoothed_labels.append(smoothed_label)
        predicts.append(batch_scores)
        loss = model.loss(batch_scores, smoothed_label, onehot=onehot)
        loss.backward(retained_graph=True)
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy_smoothing(batch_scores, smoothed_label)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)
    with open(f'train_smoothed_labels_{epoch}.csv', 'w') as f:
        f.write("GT,smoothed Label,predict\n")
        for origin, smooth, predict in zip(original_labels, smoothed_labels, predicts):
            f.write(str(origin) + "," + str(smooth) + "," + str(predict) + "\n")
    return epoch_loss, epoch_train_acc, optimizer, smoothed_labels


def smooth_evaluate_network(model, device, data_loader, epoch,  delta=1.0, onehot=False):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    smoothed_labels = []
    original_labels = []
    predicts = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_labels = batch_labels.to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            batch_scores, smoothed_label = model.forward(g=batch_graphs, h=batch_x, e=batch_e, label=batch_labels, delta=delta, snorm_e=batch_snorm_e, snorm_n=batch_snorm_n, onehot=onehot)
            original_labels.append(batch_labels)
            smoothed_labels.append(smoothed_label)
            predicts.append(batch_scores)
            loss = model.loss(batch_scores, smoothed_label, onehot)
            epoch_test_loss += loss.detach().item()
            if onehot:
                epoch_test_acc += accuracy_smoothing(batch_scores, smoothed_label)
            else:
                epoch_test_acc += accuracy(batch_scores, smoothed_label)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)
    with open(f'test_smoothed_labels_{epoch}.csv','w') as f:
        f.write("GT,smoothed Label,predict\n")
        for origin, smooth, predict in zip(original_labels, smoothed_labels, predicts):
            f.write(str(origin)+","+str(smooth)+","+str(predict)+"\n")

    return epoch_test_loss, epoch_test_acc

def train_epoch(model, optimizer, device, data_loader, epoch, onehot=False):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        batch_labels = batch_labels.to(device)
        batch_snorm_n = batch_snorm_n.to(device)         # num x 1
        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
        loss = model.loss(batch_scores, batch_labels, onehot)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        if onehot:
            epoch_train_acc += accuracy_smoothing(batch_scores, batch_labels)
        else:
            epoch_train_acc += accuracy(batch_scores, batch_labels)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)
    
    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network(model, device, data_loader, epoch, onehot=False):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_labels = batch_labels.to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
            loss = model.loss(batch_scores, batch_labels, onehot)
            epoch_test_loss += loss.detach().item()
            if onehot:
                epoch_test_acc += accuracy_smoothing(batch_scores, batch_labels)
            else:
                epoch_test_acc += accuracy(batch_scores, batch_labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)
        
    return epoch_test_loss, epoch_test_acc


