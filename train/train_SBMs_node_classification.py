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
import pickle


def smooth_train_epoch(model, optimizer, device, data_loader, epoch, lb_delta, ub_delta, train_soft_target=False):
    # test code to debug
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    batch_graph_list = []
    batch_label_list = []
    smoothed_label_list = []
    batch_scores_list = []
    weights = []
    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        batch_labels = batch_labels.to(device)
        batch_snorm_n = batch_snorm_n.to(device)  # num x 1
        optimizer.zero_grad()
        batch_scores, smoothed_label, g, saved_w = model.forward(g=batch_graphs, h=batch_x, e=batch_e, label=batch_labels,
                                                     lb_delta=lb_delta, ub_delta=ub_delta, snorm_e=batch_snorm_e, snorm_n=batch_snorm_n,
                                                     train_soft_target=train_soft_target)

        batch_label_list.append(batch_labels)
        smoothed_label_list.append(smoothed_label)
        batch_scores_list.append(batch_scores)
        batch_graph_list.append(g)
        weights.append(saved_w)
        loss = model.loss(batch_scores, smoothed_label, train_soft_target=train_soft_target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy_smoothing(batch_scores, batch_labels)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)
    return epoch_loss, epoch_train_acc, optimizer, batch_graph_list, batch_label_list, smoothed_label_list, weights, batch_scores_list


def smooth_evaluate_network(model, device, data_loader, epoch,  lb_delta, ub_delta, train_soft_target=False):
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
            batch_scores, smoothed_label, g,  saved_w = model.forward(g=batch_graphs, h=batch_x, e=batch_e, label=batch_labels, lb_delta=lb_delta, ub_delta=ub_delta, snorm_e=batch_snorm_e, snorm_n=batch_snorm_n, train_soft_target=train_soft_target)
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


