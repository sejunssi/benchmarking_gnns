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


def smooth_train_epoch(model, optimizer, device, data_loader, epoch, delta=1.0, train_soft_target=False):
    # test code to debug
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        batch_labels = batch_labels.to(device)
        batch_snorm_n = batch_snorm_n.to(device)  # num x 1
        optimizer.zero_grad()
        batch_scores, smoothed_label = model.forward(g=batch_graphs, h=batch_x, e=batch_e, label=batch_labels,
                                                     delta=delta, snorm_e=batch_snorm_e, snorm_n=batch_snorm_n,
                                                     train_soft_target=train_soft_target)
        loss = model.loss(batch_scores, smoothed_label, train_soft_target=train_soft_target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy_smoothing(batch_scores, batch_labels)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)
    return epoch_loss, epoch_train_acc, optimizer


def smooth_evaluate_network(model, device, data_loader, epoch,  delta=1.0, train_soft_target=False):
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
            batch_scores, smoothed_label = model.forward(g=batch_graphs, h=batch_x, e=batch_e, label=batch_labels, delta=delta, snorm_e=batch_snorm_e, snorm_n=batch_snorm_n, train_soft_target=train_soft_target)
            loss = model.loss(batch_scores, batch_labels.to(torch.float), train_soft_target=True)
            epoch_test_loss += loss.detach().item()
            if train_soft_target:
                epoch_test_acc += accuracy_smoothing(batch_scores, batch_labels)
            else:
                epoch_test_acc += accuracy(batch_scores, batch_labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)
    return epoch_test_loss, epoch_test_acc

def train_epoch(model, optimizer, device, data_loader, epoch, train_soft_target=False):
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
        batch_snorm_n = batch_snorm_n.to(device)  # num x 1
        optimizer.zero_grad()
        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
        loss = model.loss(batch_scores, batch_labels, train_soft_target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        if train_soft_target:
            epoch_train_acc += accuracy_smoothing(batch_scores, batch_labels)
        else:
            epoch_train_acc += accuracy(batch_scores, batch_labels)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)

    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network(model, device, data_loader, epoch, train_soft_target=False):
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
            loss = model.loss(batch_scores, batch_labels, train_soft_target)
            epoch_test_loss += loss.detach().item()
            if train_soft_target:
                epoch_test_acc += accuracy_smoothing(batch_scores, batch_labels)
            else:
                epoch_test_acc += accuracy(batch_scores, batch_labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)

    return epoch_test_loss, epoch_test_acc


