import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from util import read_dataset, mixup_data
import callbacks
from model import create_model
from torchmetrics.classification import Accuracy
import argparse
import matplotlib.pyplot as plt
from dataset import Custom_dataset
from torchmetrics.classification import Accuracy, Recall, Precision, F1Score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json


parser = argparse.ArgumentParser(description='Train a model on specific dataset')
parser.add_argument('--dataset_name', type=str, default='PAD', help='Name of the dataset: PAD, Ol3i, ODIR')
parser.add_argument('--model_str', type=str, default='image_only', help='Model type: image_only, multi_modal')
parser.add_argument('--setting', type=str, default='default', help='Setting of the dataset: sex, age, all, etc')
parser.add_argument('--result_dir', type=str, default='result_train', help='Directory to save the result')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--n_epochs', type=int, default=100, help='Number of training epochs')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader, test_loader ,val_loader = read_dataset(args.dataset_name, args.model_str, args.setting, config_path='config.json', test=False)
model = create_model(args.dataset_name, args.model_str, args.setting)
model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
n_epochs = args.n_epochs
valid_loss_min = np.Inf
accuracy = []
model_str = args.model_str
dataset_name = args.dataset_name

lr = args.lr
weight_decay = args.weight_decay
freeze = False
dt = True
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
if dt:
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)
    print('Using ReduceLROnPlateau')

acc_best = 0.
auc_best = 0.
recall_best = 0.
result_dir = args.result_dir
os.makedirs(result_dir, exist_ok=True)
loss_history = callbacks.LossHistory(result_dir)


for epoch in range(1, n_epochs + 1):
    if freeze:
        if epoch < 30:
            model.freeze_backbone()
        else:
            model.unfreeze_backbone()
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    val_pred = []
    val_target = []
    val_prob = []

    model.train()
    train_loop = tqdm(train_loader, total=len(train_loader))
    for batch in train_loop:
        if model_str == 'image_only':
            if dataset_name == 'PAD':
                img, label, _, _, _ = batch
            else:
                img, label, _, _ = batch
            img = img.to(device)
            label = label.type(torch.LongTensor).to(device)
        if model_str == 'multi_modal':
            if dataset_name == 'PAD':
                img, meta, label, _, _, _ = batch
            else:
                img, meta, label, _, _ = batch
            img = img.to(device)
            meta = meta.type(torch.FloatTensor).to(device)
            label = label.type(torch.LongTensor).to(device)

        img, targets_a, targets_b, lam = mixup_data(img, label, alpha=0.2)
        optimizer.zero_grad()
        if model_str == 'image_only':
            output = model(img=img).to(device)
        if model_str == 'multi_modal':
            output = model(img=img, meta=meta).to(device)
        loss = lam * criterion(output, targets_a) + (1 - lam) * criterion(output, targets_b)    #mixup
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_loop.set_description(f'Train Epoch [{epoch}/{n_epochs}]')

    model.eval()
    val_loop = tqdm(val_loader, total=len(val_loader))
    for batch in val_loop:
        if model_str == 'image_only':
            if dataset_name == 'PAD':
                img, label, _, _, _ = batch
            else:
                img, label, _, _ = batch
            img = img.to(device)
            label = label.type(torch.LongTensor).to(device)
        if model_str == 'multi_modal':
            if dataset_name == 'PAD':
                img, meta, label, _, _, _ = batch
            else:
                img, meta, label, _, _ = batch
            img = img.to(device)
            meta = meta.type(torch.FloatTensor).to(device)
            label = label.type(torch.LongTensor).to(device)

        with torch.no_grad():
            if model_str == 'image_only':
                output = model(img=img).to(device)
            if model_str == 'multi_modal':
                output = model(img=img, meta=meta).to(device)
        loss = criterion(output, label)
        valid_loss += loss.item()
        val_pred.append(output)
        if dataset_name == 'Ol3i':
            probabilities = torch.sigmoid(output).cpu().numpy()[:, 1]
        else:
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()
        val_target.append(label.cpu().numpy())
        val_prob.append(probabilities)
        train_loop.set_description(f'Val Epoch [{epoch}/{n_epochs}]')
    val_prob = np.concatenate(val_prob)
    val_target = np.concatenate(val_target)
    if dataset_name == 'Ol3i':
        auc = roc_auc_score(val_target, val_prob)
    else:
        auc = roc_auc_score(val_target, val_prob, average='macro', multi_class='ovr')

    # 计算平均损失
    train_loss = train_loss / len(train_loader)
    valid_loss = valid_loss / len(val_loader)
    if dt:
        scheduler.step(valid_loss)
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tAuc: {:.3f}%'.format(
        epoch, train_loss, valid_loss, auc * 100))
    with open(os.path.join(result_dir, 'result.txt'), 'a+') as file:
        file.write(
            'Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tAuc: {:.3f}%\n'.format(epoch, train_loss,
                                                                                                  valid_loss,
                                                                                                  auc * 100))
    loss_history.append_loss(epoch, 'auc', auc=auc * 100)
    loss_history.append_loss(epoch, 'loss', train_loss=train_loss, valid_loss=valid_loss)

    if auc >= auc_best:
        print('auc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(auc_best, auc))
        torch.save(model.state_dict(), os.path.join(result_dir, f'best_auc.pt'))
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        auc_best = auc
    if valid_loss <= valid_loss_min:
        print('loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), os.path.join(result_dir, f'best_loss.pt'))
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        valid_loss_min = valid_loss

