from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from glob import glob
from PIL import Image
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import os
import random
import json
import pickle


def load_transforms(transforms_config):
    transform_train = create_transform_pipeline(transforms_config['transform_train'])
    transform_test = create_transform_pipeline(transforms_config['transform_test'])
    return transform_train, transform_test


def create_transform_pipeline(transform_config):
    transform_pipeline = []
    for transform, args in transform_config.items():
        # 特殊处理 Normalize，因为它需要接收两个参数 mean 和 std
        if transform == "Normalize":
            # Normalize 期望分别为 mean 和 std 传递两个列表，确保两者都被正确传递
            mean, std = args[0][0], args[0][1]  # 根据你实际的 JSON 结构调整
            transform_instance = getattr(transforms, transform)(mean, std)
        elif isinstance(args, dict):
            # 处理其他可能以关键字参数形式提供的转换
            transform_instance = getattr(transforms, transform)(**args)
        elif isinstance(args, list):
            # 处理其他可能以位置参数形式提供的转换
            transform_instance = getattr(transforms, transform)(*args)
        elif args is None:
            # 处理无参数的转换
            transform_instance = getattr(transforms, transform)()
        else:
            raise TypeError(f"Arguments for {transform} are not properly formatted.")

        transform_pipeline.append(transform_instance)

    return transforms.Compose(transform_pipeline)
# def create_transform_pipeline(transform_config):
#     transform_pipeline = []
#     for transform, args in transform_config.items():
#         if transform == "Normalize":
#             # Normalize 期望分别为 mean 和 std 传递两个列表，确保两者都被正确传递
#             mean, std = args[0][0], args[0][1]  # 根据你实际的 JSON 结构调整
#             transform_pipeline.append(getattr(transforms, transform)(mean, std))
#         elif args is not None:
#             transform_pipeline.append(getattr(transforms, transform)(*args))
#         else:
#             transform_pipeline.append(getattr(transforms, transform)())
#     return transforms.Compose(transform_pipeline)

class Custom_dataset(Dataset):
    def __init__(self, dataset_name, model_str, setting='default', config_path='config.json', mode='train'):
        with open(config_path, 'r') as file:
            dataset_config = json.load(file)
        self.model_type = model_str
        self.dataset_name = dataset_name
        self.config = dataset_config[dataset_name]
        self.metadata_config = setting
        self.base_dir = self.config['base_dir']
        self.split = self.config['split']
        self.meta_data = pd.read_csv(self.config['metadata'])
        self.transform_train, self.transform_test = load_transforms(self.config['transforms'])

        # self.transform_train = transforms.Compose([
        #     transforms.Resize([300, 300]),
        #     transforms.CenterCrop(224),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomRotation(30),
        #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        #     transforms.ToTensor(),
        #     transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        #
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        #
        # self.transform_test = transforms.Compose([
        #     transforms.Resize([224, 224]),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        self.mode = mode
        if self.mode == 'train':
            self.images = pd.read_csv(self.split)['train']
        elif self.mode == 'test':
            self.images = pd.read_csv(self.split)['test']
        elif self.mode == 'val':
            self.images = pd.read_csv(self.split)['val']
        self.images = self.images.dropna(axis=0, how='any')

        self.sex_id_matrix = np.eye(3)
        self.smoke_id_matrix = np.eye(3)
        self.drink_id_matrix = np.eye(3)
        self.background_father_id_matrix = np.eye(14)   #6-20
        self.background_mother_id_matrix = np.eye(12)   #20-32
        self.pesticide_id_matrix = np.eye(3)
        # self.gender_id_matrix = np.eye(3) #35-38
        self.skin_cancer_history_id_matrix = np.eye(3)  #38-41
        self.cancer_history_id_matrix = np.eye(3)   #41-44
        self.has_piped_water_id_matrix = np.eye(3)  #44-47
        self.has_sewage_system_id_matrix = np.eye(3)    #47-50
        self.region_id_matrix = np.eye(15)  #50-65
        self.itch_id_matrix = np.eye(3) #65-68
        self.grew_id_matrix = np.eye(3) #68-71
        self.hurt_id_matrix = np.eye(3) #71-74
        self.changed_id_matrix = np.eye(3)  #74-77
        self.bleed_id_matrix = np.eye(3)    #77-80
        self.elevation_id_matrix = np.eye(3)    #80-83
        self.biopsed_id_matric = np.eye(3)  #83-86
        self.fitspatrick_id_matrix = np.eye(7)  #86-93
        self.cls_ids_prep()

    def cls_ids_prep(self):
        if self.dataset_name == 'PAD': num_classes = 6
        if self.dataset_name == 'Ol3i': num_classes = 2
        if self.dataset_name == 'ODIR': num_classes = 8
        self.img_cls_ids = [[] for i in range(num_classes)]
        for i in range(num_classes):
            self.img_cls_ids[i] = np.where(self.meta_data['label'] == i)[0]

    def getlabel(self, idx):
        img_name = self.images[idx]
        label = self.meta_data.loc[self.meta_data['image'] == img_name]['label'].values[0]
        return label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.base_dir, self.images[idx])).convert('RGB')
        if self.mode == 'train':
            image = self.transform_train(image)
        elif self.mode == 'test':
            image = self.transform_test(image)
        elif self.mode == 'val':
            image = self.transform_test(image)

        ori_label = self.meta_data.loc[self.meta_data['image'] == img_name]['label'].values[0]

        row = self.meta_data.loc[self.meta_data['image'] == img_name]
        ori_sex = row['sex'].values[0]
        ori_age = row['age'].values[0]
        sex = self.sex_id_matrix[row['sex'].values[0]]
        # print(age)
        age_min, age_max = 0, 100
        normalized_age = (ori_age - age_min) / (ori_age - age_min)
        age = normalized_age
        ori_label = torch.tensor(ori_label)
        if self.model_type == 'image_only':
            if self.dataset_name == 'PAD':
                ori_skin = row['fitspatrick_num'].values[0]
                return image, ori_label, ori_sex, ori_age, ori_skin
            else:
                return image, ori_label, ori_sex, ori_age
        elif self.model_type == 'multi_modal':
            if self.dataset_name == 'PAD':
                ori_skin = row['fitspatrick_num'].values[0]
                smoke = self.smoke_id_matrix[row['smoke_num'].values[0]]  # 1
                drink = self.drink_id_matrix[row['drink_num'].values[0]]
                background_father = self.background_father_id_matrix[row['background_father_num'].values[0]]
                background_mother = self.background_mother_id_matrix[row['background_mother_num'].values[0]]
                pesticide = self.pesticide_id_matrix[row['pesticide_num'].values[0]]  # 5
                # gender = self.gender_id_matrix[row['gender_num'].values[0]]
                skin_cancer_history = self.skin_cancer_history_id_matrix[row['skin_cancer_history_num'].values[0]]
                cancer_history = self.cancer_history_id_matrix[row['cancer_history_num'].values[0]]  # 8
                has_piped_water = self.has_piped_water_id_matrix[row['has_piped_water_num'].values[0]]
                has_sewage_system = self.has_sewage_system_id_matrix[row['has_sewage_system_num'].values[0]]  # 10
                region = self.region_id_matrix[row['region_num'].values[0]]
                itch = self.itch_id_matrix[row['itch_num'].values[0]]  # 12
                grew = self.grew_id_matrix[row['grew_num'].values[0]]
                hurt = self.hurt_id_matrix[row['hurt_num'].values[0]]
                changed = self.changed_id_matrix[row['changed_num'].values[0]]  # 15
                bleed = self.bleed_id_matrix[row['bleed_num'].values[0]]
                elevation = self.elevation_id_matrix[row['elevation_num'].values[0]]
                biopsed = self.biopsed_id_matric[row['biopsed_num'].values[0]]
                fitspatrick = self.fitspatrick_id_matrix[row['fitspatrick_num'].values[0]]
                diameter_1 = row['diameter_1'].values[0]
                diameter_2 = row['diameter_2'].values[0]
                diameter_1_min, diameter_1_max = 0, 100
                diameter_2_min, diameter_2_max = 0, 100
                normalized_diameter_1 = (diameter_1 - diameter_1_min) / (diameter_1_max - diameter_1_min)
                normalized_diameter_2 = (diameter_2 - diameter_2_min) / (diameter_2_max - diameter_2_min)
                diameter_1 = normalized_diameter_1
                diameter_2 = normalized_diameter_2
                if self.metadata_config == 'age':
                    meta_data = age
                    meta_data = torch.tensor(meta_data)
                    meta_data = meta_data.unsqueeze(0)
                elif self.metadata_config == 'sex':
                    meta_data = sex
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'skin':
                    meta_data = fitspatrick
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'without_smoke':
                    meta_data = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                        np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                            np.append(np.append(np.append(drink, background_father), background_mother),
                                      pesticide), sex), skin_cancer_history), cancer_history), region),
                            has_piped_water), has_sewage_system), itch), grew), hurt), changed), bleed),
                        elevation), biopsed), fitspatrick),
                        age), diameter_1), diameter_2)
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'without_drink':
                    meta_data = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                        np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                            np.append(np.append(np.append(smoke, background_father), background_mother),
                                      pesticide), sex), skin_cancer_history), cancer_history), region),
                            has_piped_water), has_sewage_system), itch), grew), hurt), changed), bleed),
                        elevation), biopsed), fitspatrick),
                        age), diameter_1), diameter_2)
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'without_father_and_mother_background':
                    meta_data = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                        np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                            np.append(np.append(np.append(smoke, drink),
                                      pesticide), sex), skin_cancer_history), cancer_history), has_piped_water),
                                                      has_sewage_system), region), itch), grew), hurt), changed), bleed),
                                                                                            elevation), biopsed), fitspatrick),
                                                              age), diameter_1), diameter_2)
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'without_pesticide':
                    meta_data = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                        np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                            np.append(np.append(np.append(smoke, drink), background_father),
                                      background_mother), sex), skin_cancer_history), cancer_history), region),
                            has_piped_water), has_sewage_system), itch), grew), hurt), changed), bleed),
                        elevation), biopsed), fitspatrick),
                        age), diameter_1), diameter_2)
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'without_skin_cancer_history':
                    meta_data = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                        np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                            np.append(np.append(np.append(smoke, drink), background_father),
                                      background_mother), pesticide), sex), cancer_history), region),
                            has_piped_water), has_sewage_system), itch), grew), hurt), changed), bleed),
                        elevation), biopsed), fitspatrick),
                        age), diameter_1), diameter_2)
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'without_cancer_history':
                    meta_data = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                        np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                            np.append(np.append(np.append(smoke, drink), background_father),
                                      background_mother), pesticide), sex), skin_cancer_history), region),
                            has_piped_water), has_sewage_system), itch), grew), hurt), changed), bleed),
                        elevation), biopsed), fitspatrick),
                        age), diameter_1), diameter_2)
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'without_piped_water_and_sewage_system':
                    meta_data = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                        np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                            np.append(np.append(np.append(smoke, drink), #background_mother),
                                      background_father), background_mother), pesticide), sex), skin_cancer_history),
                                                      cancer_history), region), itch), grew), hurt), changed), bleed),
                                                                                            elevation), biopsed), fitspatrick),
                                                              age), diameter_1), diameter_2)
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'without_region':
                    meta_data = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                        np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                            np.append(np.append(np.append(smoke, drink), background_father),
                                      background_mother), pesticide), sex), skin_cancer_history), cancer_history),
                            has_piped_water), has_sewage_system), itch), grew), hurt), changed), bleed),
                        elevation), biopsed), fitspatrick),
                        age), diameter_1), diameter_2)
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'without_itch':
                    meta_data = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                        np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                            np.append(np.append(np.append(smoke, drink), background_father),
                                      background_mother), pesticide), sex), skin_cancer_history), cancer_history),
                            has_piped_water), has_sewage_system), region), grew), hurt), changed), bleed),
                        elevation), biopsed), fitspatrick),
                        age), diameter_1), diameter_2)
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'without_grew':
                    meta_data = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                        np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                            np.append(np.append(np.append(smoke, drink), background_father),
                                      background_mother), pesticide), sex), skin_cancer_history), cancer_history),
                            has_piped_water), has_sewage_system), region), itch), hurt), changed), bleed),
                        elevation), biopsed), fitspatrick),
                        age), diameter_1), diameter_2)
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'without_hurt':
                    meta_data = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                        np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                            np.append(np.append(np.append(smoke, drink), background_father),
                                      background_mother), pesticide), sex), skin_cancer_history), cancer_history),
                            has_piped_water), has_sewage_system), region), itch), grew), changed), bleed),
                        elevation), biopsed), fitspatrick),
                        age), diameter_1), diameter_2)
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'without_changed':
                    meta_data = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                        np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                            np.append(np.append(np.append(smoke, drink), background_father),
                                      background_mother), pesticide), sex), skin_cancer_history), cancer_history),
                            has_piped_water), has_sewage_system), region), itch), grew), hurt), bleed),
                        elevation), biopsed), fitspatrick),
                        age), diameter_1), diameter_2)
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'without_bleed':
                    meta_data = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(smoke, drink), background_father),background_mother), pesticide), sex), skin_cancer_history), cancer_history),has_piped_water), has_sewage_system), region), itch), grew), hurt), changed),elevation), biopsed), fitspatrick),age), diameter_1), diameter_2)
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'without_elevation':
                    meta_data = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                        np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                            np.append(np.append(np.append(smoke, drink), background_father),
                                      background_mother), pesticide), sex), skin_cancer_history), cancer_history),
                            has_piped_water), has_sewage_system), region), itch), grew), hurt), changed),
                        bleed), biopsed), fitspatrick),
                        age), diameter_1), diameter_2)
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'without_biopsed':
                    meta_data = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                        np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                            np.append(np.append(np.append(smoke, drink), background_father),
                                      background_mother), pesticide), sex), skin_cancer_history), cancer_history),
                            has_piped_water), has_sewage_system), region), itch), grew), hurt), changed),
                        bleed), elevation), fitspatrick),
                        age), diameter_1), diameter_2)
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'without_fitzpatrick':
                    meta_data = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                        np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                            np.append(np.append(np.append( smoke, drink), background_father),
                                      background_mother), pesticide), sex), skin_cancer_history), cancer_history),
                                                      has_piped_water), has_sewage_system), region), itch), grew), hurt), changed),
                                                                                            bleed), elevation), biopsed),
                                                              age), diameter_1), diameter_2)
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'without_age':
                    meta_data = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                        np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                            np.append(np.append(np.append(smoke, drink), background_father),
                                      background_mother), pesticide), sex), skin_cancer_history), cancer_history),
                            has_piped_water), has_sewage_system), region), itch), grew), hurt), changed),
                        bleed), elevation), biopsed),
                        fitspatrick), diameter_1), diameter_2)
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'without_diameter':
                    meta_data = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                        np.append(np.append(np.append(np.append(np.append(np.append(np.append(
                            np.append(np.append(np.append(smoke, drink), #background_mother),
                                      background_father), background_mother), pesticide), sex), skin_cancer_history),
                                                      cancer_history), has_piped_water), has_sewage_system), region), itch), grew), hurt),
                                                                                            changed), bleed), elevation),
                                                              biopsed), fitspatrick), age)
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'all' or self.metadata_config == 'default':
                    meta_data = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(smoke,drink),background_father),background_mother),pesticide),sex),skin_cancer_history),cancer_history),has_piped_water),has_sewage_system),region),itch),grew),hurt),changed),bleed),elevation),biopsed),fitspatrick),age),diameter_1),diameter_2)
                    meta_data = torch.tensor(meta_data)
                return image, meta_data, ori_label, ori_sex, ori_age, ori_skin
            else:
                if self.metadata_config == 'age':
                    meta_data = age
                    meta_data = torch.tensor(meta_data)
                    meta_data = meta_data.unsqueeze(0)
                elif self.metadata_config == 'sex':
                    meta_data = sex
                    meta_data = torch.tensor(meta_data)
                elif self.metadata_config == 'all' or self.metadata_config == 'default':
                    meta_data = np.append(age, sex)
                    meta_data = torch.tensor(meta_data)
                return image, meta_data, ori_label, ori_sex, ori_age





