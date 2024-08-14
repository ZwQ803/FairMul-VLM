import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from util import read_dataset
from model import create_model
from torchmetrics.classification import Accuracy, Recall, Precision, F1Score
import argparse
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
torch.autograd.set_detect_anomaly(True)


# 给这个解析对象添加命令行参数
parser = argparse.ArgumentParser(description='Test the pretrained models and save the results.')
parser.add_argument('--dataset_name', type=str, default='PAD', help='Name of the dataset: PAD, Ol3i, ODIR')
parser.add_argument('--model_str', type=str, default='image_only', help='Model type: image_only, multi_modal')
parser.add_argument('--setting', type=str, default='default', help='Setting of the dataset: sex, age, all, etc')
parser.add_argument('--result_dir', type=str, default='result_eval', help='Directory to save the result')
parser.add_argument('--model_file', type=str, default='result_train/best_auc.pt', help='Path to the pretrained model file.')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader, test_loader ,val_loader = read_dataset(args.dataset_name, args.model_str, args.setting, config_path='config.json', test=False)
model = create_model(args.dataset_name, args.model_str, args.setting)
model = model.to(device)
model_str = args.model_str
dataset_name = args.dataset_name
pretrained_state_dict = torch.load(args.model_file)
if dataset_name == 'PAD':
    cls = {'ACK': 0, 'BCC': 1, 'MEL': 2, 'NEV': 3, 'SCC': 4, 'SEK': 5}
    num_classes = 6
elif dataset_name == 'ODIR':
    cls = {'N': 0, 'D': 1, 'G': 2, 'C': 3, 'A': 4, 'H': 5, 'M': 6, 'O': 7}
    num_classes = 8
elif dataset_name == 'Ol3i':
    cls = {'N': 0, 'P': 1}
    num_classes = 2

result_dir = args.result_dir
os.makedirs(result_dir, exist_ok=True)
# 从预训练状态字典中删除 'meta_input'
pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k != "meta_input"}
model.load_state_dict(pretrained_state_dict)



all_val_pred = []
all_val_target = []
all_prob = []
# 创建两个空列表，用于分别存储男性和女性的预测结果和真实标签

male_val_pred = []
male_val_target = []
male_prob = []
female_val_pred = []
female_val_target = []
female_prob = []
# 创建两个空列表，用于分别存储老年人和青年人的预测结果和真实标签

old_val_pred = []
old_val_target = []
old_prob = []
young_val_pred = []
young_val_target = []
young_prob = []

light_val_pred = []
light_val_target = []
light_prob = []
dark_val_pred = []
dark_val_target = []
dark_prob = []



model.eval()  # 验证模型
test_loop = tqdm(test_loader, total=len(test_loader))

idx = 0
# for img, meta, label in teat_loop:
for batch in test_loop:
    if model_str == 'image_only':
        if dataset_name == 'PAD':
            img, label, sex, age, skin = batch
        else:
            img, label, sex, age = batch
        img = img.to(device)
        label = label.type(torch.LongTensor).to(device)
    if model_str == 'multi_modal':
        if dataset_name == 'PAD':
            img, meta, label, sex, age, skin = batch
        else:
            img, meta, label, sex, age = batch
        img = img.to(device)
        meta = meta.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)

    with torch.no_grad():
        if model_str == 'image_only':
            output = model(img=img).to(device)
        if model_str == 'multi_modal':
            output = model(img=img, meta=meta).to(device)
    _, pred = torch.max(output, 1)
    if dataset_name == 'Ol3i':
        probabilities = torch.sigmoid(output).cpu().numpy()[:, 1]
    else:
        probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()
    # 将所有测试数据的预测结果和真实标签添加到列表中
    all_val_pred.append(pred)
    all_val_target.append(label)
    all_prob.append(probabilities)
    # print("all_prob",all_prob)
    for i in range(len(img)):
        if age[i] > 64:  # >65岁
            old_val_pred.append(pred[i].item())
            old_val_target.append(label[i].item())
            old_prob.append(probabilities[i])
        # elif torch.lt(age_info, torch.tensor([0.18], device=device)):  # <18岁
        else:
            young_val_pred.append(pred[i].item())
            young_val_target.append(label[i].item())
            young_prob.append(probabilities[i])
            # print(young_prob)

        if sex[i] == 2:  # nvxing
            male_val_pred.append(pred[i].item())
            male_val_target.append(label[i].item())
            male_prob.append(probabilities[i])
        elif sex[i] == 1:  # 男性 这里写反里，输出先female再male，先男再女
            female_val_pred.append(pred[i].item())
            female_val_target.append(label[i].item())
            female_prob.append(probabilities[i])
        if dataset_name == 'PAD':
            if skin[i] == 1 or skin[i] == 2 or skin[i] == 3:
                light_val_pred.append(pred[i].item())
                light_val_target.append(label[i].item())
                light_prob.append(probabilities[i])
            elif skin[i] == 4 or skin[i] == 5 or skin[i] == 6:
                dark_val_pred.append(pred[i].item())
                dark_val_target.append(label[i].item())
                dark_prob.append(probabilities[i])


# 合并所有测试数据的预测结果和真实标签
all_val_pred = torch.cat(all_val_pred, dim=0)
all_val_target = torch.cat(all_val_target, dim=0)
all_prob = np.concatenate(all_prob)

male_val_pred = torch.tensor(male_val_pred)
female_val_pred = torch.tensor(female_val_pred)
male_val_target = torch.tensor(male_val_target)
female_val_target = torch.tensor(female_val_target)
# 合并男性和女性的预测结果和真实标签
male_val_pred = torch.cat((male_val_pred,), dim=0)
female_val_pred = torch.cat((female_val_pred,), dim=0)
male_val_target = torch.cat((male_val_target,), dim=0)
female_val_target = torch.cat((female_val_target,), dim=0)

old_val_pred = torch.tensor(old_val_pred)
young_val_pred = torch.tensor(young_val_pred)
old_val_target = torch.tensor(old_val_target)
young_val_target = torch.tensor(young_val_target)

# 合并年龄的预测结果和真实标签
old_val_pred = torch.cat((old_val_pred,), dim=0)
young_val_pred = torch.cat((young_val_pred,), dim=0)
old_val_target = torch.cat((old_val_target,), dim=0)
young_val_target = torch.cat((young_val_target,), dim=0)

if dataset_name == 'PAD':
    light_val_pred = torch.tensor(light_val_pred)
    dark_val_pred = torch.tensor(dark_val_pred)
    light_val_target = torch.tensor(light_val_target)
    dark_val_target = torch.tensor(dark_val_target)

    light_val_target = torch.cat((light_val_target,), dim=0)
    light_val_pred = torch.cat((light_val_pred,), dim=0)
    dark_val_pred = torch.cat((dark_val_pred,), dim=0)
    dark_val_target = torch.cat((dark_val_target,), dim=0)

# 生成热力图并保存
def save_heatmap(val_cm, heatmap_result_dir):
    xtick = cls.keys()
    ytick = cls.keys()
    # 处理分母为零的情况，将分母为零的行的所有元素设置为0，以避免除法错误
    plt.clf()
    # val_cm = np.where(val_cm.sum(axis=1)[:, np.newaxis] == 0, 0, val_cm)
    val_cm = val_cm.astype('float') / val_cm.sum(axis=1)[:, np.newaxis]
    h = sns.heatmap(val_cm, fmt='.2f', cmap="Blues", annot=True, cbar=False, xticklabels=xtick, yticklabels=ytick)
    cb = h.figure.colorbar(h.collections[0])
    plt.savefig(os.path.join(heatmap_result_dir, 'confusionmatrix.png'))

def metrics_to_csv(val_target, val_pred, prob_np, metrics_result_dir, tag):
    # 保存混淆矩阵为CSV文件，文件名中包含准确率
    acc = accuracy_score(val_target.cpu().numpy(), val_pred.cpu().numpy())
    recall = recall_score(val_target.cpu().numpy(), val_pred.cpu().numpy(), average='macro', zero_division=1)
    precision = precision_score(val_target.cpu().numpy(), val_pred.cpu().numpy(), average='macro', zero_division=1)
    # f1_macro = f1_score(val_target.cpu().numpy(), val_pred.cpu().numpy(), average='macro')
    f1_micro = f1_score(val_target.cpu().numpy(), val_pred.cpu().numpy(), average='micro')
    # 提取概率分数并转换为 NumPy 数组
    # prob_np = np.concatenate([prob.cpu().numpy() for prob in val_prob])
    # prob_np = prob_np[:, 1]
    # print(val_target.cpu().numpy().shape, prob_np.shape)
    auc = roc_auc_score(val_target.cpu().numpy(), prob_np, average='macro', multi_class='ovr')
    # 创建一个包含指标名称和对应值的列表
    metric_names = ["Accuracy", "recall", "Precision", "F1 Micro"]  # , "AUC"]
    metric_values = [acc.item(), recall.item(), precision.item(), f1_micro.item(), auc.item()]  # , auc.item()]
    val_cm = confusion_matrix(val_target.cpu().numpy(),val_pred.cpu().numpy())
    # 将指标名称和对应值保存到 CSV 文件

    with open(os.path.join(metrics_result_dir, "confusion_matrix.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(val_cm)
    save_heatmap(val_cm, metrics_result_dir)
    return metric_values

accuracy_metric = Accuracy(task="multiclass",num_classes=num_classes).to(device)
all_acc = accuracy_metric(all_val_pred, all_val_target)
result_dir_all = os.path.join(result_dir, f'logs_all_acc={all_acc:.2f}')
os.makedirs(result_dir_all, exist_ok=True)
all_metric_values = metrics_to_csv(all_val_target, all_val_pred, all_prob, result_dir_all, tag="all")


male_acc = accuracy_metric(male_val_target, male_val_pred)
female_acc = accuracy_metric(female_val_target, female_val_pred)
result_dir_male = os.path.join(result_dir, f'logs_male_acc={male_acc:.2f}')
result_dir_female = os.path.join(result_dir, f'logs_female_acc={female_acc:.2f}')
os.makedirs(result_dir_male, exist_ok=True)
os.makedirs(result_dir_female, exist_ok=True)
female_metric_values = metrics_to_csv(female_val_target,female_val_pred,female_prob,result_dir_female,tag="female")
male_metric_values = metrics_to_csv(male_val_target, male_val_pred, male_prob, result_dir_male, tag="male")


old_acc = accuracy_metric(old_val_target, old_val_pred)
young_acc = accuracy_metric(young_val_target, young_val_pred)
result_dir_old = os.path.join(result_dir, f'logs_old_acc={old_acc:.2f}')
result_dir_young = os.path.join(result_dir, f'logs_young_acc={young_acc:.2f}')
os.makedirs(result_dir_old, exist_ok=True)
os.makedirs(result_dir_young, exist_ok=True)
old_metric_values = metrics_to_csv(old_val_target, old_val_pred, old_prob, result_dir_old, tag="old")
young_metric_values = metrics_to_csv(young_val_target, young_val_pred, young_prob, result_dir_young, tag="young")

metrics = ["Accuracy", "Recall", "Precision", "F1 Micro", "AUC-ROC"]

if dataset_name == 'PAD':
    light_acc = accuracy_metric(light_val_target, light_val_pred)
    dark_acc = accuracy_metric(dark_val_target, dark_val_pred)
    result_dir_light = os.path.join(result_dir, f'logs_light_acc={light_acc:.2f}')
    result_dir_dark = os.path.join(result_dir, f'logs_dark_acc={dark_acc:.2f}')
    os.makedirs(result_dir_light, exist_ok=True)
    os.makedirs(result_dir_dark, exist_ok=True)
    light_metric_values = metrics_to_csv(light_val_target, light_val_pred, light_prob, result_dir_light, tag="light")
    dark_metric_values = metrics_to_csv(dark_val_target, dark_val_pred, dark_prob, result_dir_dark, tag="dark")

    categories = ['All', 'Female', 'Male', 'Old', 'Young', 'Light', 'Dark']
    all_data = [all_metric_values, female_metric_values, male_metric_values, old_metric_values, young_metric_values, light_metric_values, dark_metric_values]

else:
    categories = ['All', 'Female', 'Male', 'Old', 'Young']
    all_data = [all_metric_values, female_metric_values, male_metric_values, old_metric_values, young_metric_values]



# 输出 CSV 文件路径
csv_file_path = os.path.join(result_dir, "metrics.csv")

# 写入数据到 CSV 文件
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 写入头部，即指标名称
    writer.writerow(['Category'] + metrics)

    # 循环写入每个类别的指标数据
    for category, values in zip(categories, all_data):
        writer.writerow([category] + values)

print(f"Metrics written successfully to {csv_file_path}")