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


def smooth_train_epoch(model, optimizer, device, data_loader, epoch,  delta=1.0, smooth=False):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    smoothed_labels = []
    original_labels = []
    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
        # one_hot_batch_labels = [ torch.nn.functional.one_hot(label.to(torch.int64)) for label in batch_labels]
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        batch_labels = batch_labels.to(device)
        batch_snorm_n = batch_snorm_n.to(device)  # num x 1
        batch_score, smoothed_label = model.forward(g=batch_graphs, h=batch_x, e=batch_e, label=batch_labels, delta=delta, snorm_e=batch_snorm_e, snorm_n=batch_snorm_n, smooth=smooth)
        original_labels.append(original_labels)
        smoothed_labels.append(smoothed_label)
        loss = model.loss(batch_score, smoothed_label, smooth=True)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy_smoothing(batch_score, smoothed_label)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)
    with open(f'smoothed_labels.csv') as f:
        for origin, smooth in zip(original_labels, smoothed_labels):
            f.write(str(origin)+","+str(smooth)+"\n")

    return epoch_loss, epoch_train_acc, optimizer


def train_epoch(model, optimizer, device, data_loader, epoch, smooth=False):

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
        loss = model.loss(batch_scores, batch_labels, smooth)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        if smooth:
            epoch_train_acc += accuracy_smoothing(batch_scores, batch_labels)
        else:
            epoch_train_acc += accuracy(batch_scores, batch_labels)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)
    
    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network(model, device, data_loader, epoch, smooth=False):
    
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
            loss = model.loss(batch_scores, batch_labels, smooth)
            epoch_test_loss += loss.detach().item()
            if smooth:
                epoch_test_acc += accuracy_smoothing(batch_scores, batch_labels)
            else:
                epoch_test_acc += accuracy(batch_scores, batch_labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)
        
    return epoch_test_loss, epoch_test_acc


