import torch
import torch.nn as nn
from torchvision.models import swin_s
from torchvision.ops import MLP
import json

def create_model(dataset_name, model_str, setting='default', config_path='config.json'):
    with open(config_path, 'r') as file:
        config = json.load(file)
        dataset_config = config[dataset_name][model_str]
        model_params = dataset_config['settings'][setting]
        model_class = globals()[dataset_config['model_type']]
        return model_class(**model_params)


def MLP_Block(dim1, dim2, dropout=0.25):
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ReLU(),
        nn.AlphaDropout(p=dropout, inplace=False))


# PAD multi-modal fusion model
class PADMultiModalModel(nn.Module):
    def __init__(self, num_labels=6, metadata_length=96):
        super(PADMultiModalModel, self).__init__()
        meta_hidden = [64, 64, 128]
        meta_net = [MLP_Block(dim1=metadata_length, dim2=meta_hidden[0])]
        for i, _ in enumerate(meta_hidden[1:]):
            meta_net.append(MLP_Block(dim1=meta_hidden[i], dim2=meta_hidden[i + 1], dropout=0.25))
        meta_net.append(nn.Linear(in_features=128, out_features=128, bias=True))
        self.meta_net = nn.Sequential(*meta_net)
        self.swin = swin_s(weights='IMAGENET1K_V1')
        self.swin.head = nn.Linear(in_features=768, out_features=256, bias=True)
        self.fusion_net = nn.Sequential(*[nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()])
        self.classifier = nn.Linear(256, num_labels)
        self.meta_input = None
        self.meta_grads = None

    def freeze_backbone(self):
        backbone = self.swin
        for param in backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        backbone = self.swin
        for param in backbone.parameters():
            param.requires_grad = True

    def forward(self, **kwargs):
        img = kwargs['img']
        meta = kwargs['meta']
        if self.meta_input is None or self.meta_input.shape[0] != meta.shape[0]:
            self.meta_input = nn.Parameter(meta.clone(), requires_grad=True)
        else:
            self.meta_input.data = meta.clone()
        if self.meta_input.grad is None:
            self.meta_input.register_hook(lambda grad: self.save_meta_grad(grad))

        img_embed = self.swin(img).squeeze()
        meta_embed = self.meta_net(self.meta_input)
        hidden_embed = self.fusion_net(torch.cat((img_embed, meta_embed), dim=1))
        logits = self.classifier(hidden_embed)
        return logits

    def save_meta_grad(self, grad):
        self.meta_grads = grad



# PAD image only
class PADImageOnlyModel(nn.Module):
    def __init__(self, num_labels = 6):
        super(PADImageOnlyModel, self).__init__()
        self.swin = swin_s(weights='IMAGENET1K_V1')
        self.swin.head = nn.Linear(in_features=768, out_features=256, bias=True)
        self.fusion_net = nn.Sequential(*[nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()])
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, **kwargs):
        img = kwargs['img']
        img_embed = self.swin(img).squeeze()
        logits = self.classifier(self.fusion_net(img_embed))
        return logits

# Ol3i multi-modal fusion model
class Ol3iMultiModalModel(nn.Module):
    def __init__(self, num_labels=2, metadata_length=4):
        super(Ol3iMultiModalModel, self).__init__()
        self.swin = swin_s(weights='IMAGENET1K_V1')
        self.swin.head = nn.Identity()
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(768, 512)
        self.bn1 = nn.GroupNorm(32, 512)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 256)
        meta_hidden = [64, 64, 128]
        meta_net = [MLP_Block(dim1=metadata_length, dim2=meta_hidden[0])]
        for i, _ in enumerate(meta_hidden[1:]):
            meta_net.append(MLP_Block(dim1=meta_hidden[i], dim2=meta_hidden[i+1], dropout=0.1))
        meta_net.append(nn.Linear(in_features=meta_hidden[-1], out_features=128, bias=True))
        self.meta_net = nn.Sequential(*meta_net)
        self.fusion_net = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, **kwargs):
        img = kwargs['img']
        meta = kwargs['meta']
        img_embed = self.swin(img)
        img_embed = self.dropout(img_embed)
        img_embed = self.fc1(img_embed)
        img_embed = self.bn1(img_embed)
        img_embed = self.relu(img_embed)
        img_embed = self.fc2(img_embed)
        meta_embed = self.meta_net(meta)
        hidden_embed = self.fusion_net(torch.cat((img_embed, meta_embed), dim=1))
        logits = self.classifier(hidden_embed)
        return logits


# Ol3i image only
class Ol3iImageOnlyModel(nn.Module):
    def __init__(self, num_labels=2):
        super(Ol3iImageOnlyModel, self).__init__()
        self.swin = swin_s(weights='IMAGENET1K_V1')
        self.swin.head = nn.Identity()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(768, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, num_labels)

    def forward(self, **kwargs):
        x = kwargs['img']
        x = self.swin(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ODIR multi-modal fusion model
class ODIRMultiModalModel(nn.Module):
    def __init__(self, num_labels = 8, metadata_length=4):
        super(ODIRMultiModalModel, self).__init__()
        meta_hidden = [64, 64, 128]
        meta_net = [MLP_Block(dim1=metadata_length, dim2=meta_hidden[0])]
        for i, _ in enumerate(meta_hidden[1:]):
            meta_net.append(MLP_Block(dim1=meta_hidden[i], dim2=meta_hidden[i+1], dropout=0.25))
        meta_net.append(nn.Linear(in_features=128, out_features=128, bias=True))
        self.meta_net = nn.Sequential(*meta_net)
        self.swin = swin_s(weights='IMAGENET1K_V1')
        self.swin.head = nn.Linear(in_features=768, out_features=256, bias=True)
        self.fusion_net = nn.Sequential(*[nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()])
        self.classifier = nn.Linear(256, num_labels)
        self.meta_input = None
        self.meta_grads = None

    def freeze_backbone(self):
        backbone = self.swin
        for param in backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        backbone = self.swin
        for param in backbone.parameters():
            param.requires_grad = True

    def forward(self, **kwargs):
        img = kwargs['img']
        #multi
        meta = kwargs['meta']
        if self.meta_input is None or self.meta_input.shape[0] != meta.shape[0]:
            self.meta_input = nn.Parameter(meta.clone(), requires_grad=True)
        else:
            self.meta_input.data = meta.clone()

        # Add a hook to capture the gradients of meta_input
        if self.meta_input.grad is None:
            self.meta_input.register_hook(lambda grad: self.save_meta_grad(grad))

        img_embed = self.swin(img).squeeze()
        meta_embed = self.meta_net(self.meta_input)  # Use self.meta_input here instead of meta
        hidden_embed = self.fusion_net(torch.cat((img_embed, meta_embed), dim=1))
        logits = self.classifier(hidden_embed)
        return logits

    def save_meta_grad(self, grad):
        self.meta_grads = grad

# ODIR image only
class ODIRImageOnlyModel(nn.Module):
    def __init__(self, num_labels = 8):
        super(ODIRImageOnlyModel, self).__init__()
        self.swin = swin_s(weights='IMAGENET1K_V1')
        self.fusion_net = nn.Sequential(*[nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()])
        self.classifier = nn.Linear(256, num_labels)


    def freeze_backbone(self):
        backbone = self.swin
        for param in backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        backbone = self.swin
        for param in backbone.parameters():
            param.requires_grad = True

    def forward(self, **kwargs):
        img = kwargs['img']
        img_embed = self.swin(img).squeeze()
        logits = self.classifier(self.fusion_net(img_embed))
        return logits

    def save_meta_grad(self, grad):
        self.meta_grads = grad