# The Guideline for Building Fair Multimodal Medical AI with Large Vision-Language Model

This repository hosts the official implementation of the methods discussed in our paper "The Guideline for Building Fair Multimodal Medical AI with Large Vision-Language Model". The study extends fairness research to multimodal medical AI, focusing on dermatology, radiology, and ophthalmology, and investigates how patient metadata interacts with medical imaging to affect fairness.

## Prepare the environment

1. Clone this repository and navigate to FairMul-VLM folder

```bash
git clone https://github.com/ZwQ803/FairMul-VLM.git
cd FairMul-VLM
```

2. Install Package: Create conda environment

```bash
conda create -n FairMul-VLM python=3.8 -y
conda activate FairMul-VLM
pip install -r requirement.txt
```

3. Download the pre-training model

   You can download the pre-training model and fine-tuning models from [Zenodo](https://zenodo.org/records/13317382) or [baiduDisk code:k396](https://pan.baidu.com/s/1RodlZAkh_cv3es9VJFs-KA?pwd=k396). Then, you can unzip the file and put the folder `pretrained_model`  in the root directory of FairMul-VLM.

## Prepare the datasets

### 1. Downloading the Datasets

You can download the required datasets using the following links provided in our paper:

- **PAD-UFES-20**: [Access here](https://data.mendeley.com/datasets/zr7vgbcyr2/1)
- **Ol3i**: [Access here](https://stanfordaimi.azurewebsites.net/datasets/3263e34a-252e-460f-8f63-d585a9bfecfc)
- **ODIR**: [Access here](https://academictorrents.com/details/cf3b8d5ecdd4284eb9b3a80fcfe9b1d621548f72)

After downloading, unzip each dataset and place the image folders in the `Dataset` directory under the FairMul-VLM project folder.

### 2. Preparing the Dataset Splits

Follow the methodology described in our paper to split the datasets into training, validation, and testing sets. Generate a `split.csv` file for each dataset, detailing the image names in each subset. Here is an example structure for the ODIR dataset:

```
Dataset/
    ODIR/
        ODIR_split.csv
        ODIR_metadata.csv
```

### 3. Using Standard Splits

If you wish to use the dataset splits as per our paper, download the necessary `.csv` files from [our repository](https://github.com/ZwQ803/FairMul-VLM/tree/main/Dataset) and place them in the root `Dataset` directory. Remember to adjust file paths in these `.csv` files to match your local setup and update the `base_dir`, `split`, and `metadata` paths in `config.json` accordingly.

## Training and evalutaion for FairMul-VLM

### 1. Training models

#### Training Commands

For training a model with specific metadata settings, use the following command template:

```bash
python train.py --dataset_name <dataset> --model_str <model_type> --setting <metadata_setting> --lr <learning_rate> --weight_decay <weight_decay> --n_epochs <n> --result_dir <train_dir>
```

- `<dataset>`: Choose from `PAD`, `Ol3i`, or `ODIR`.
- `<model_type>`: Choose from `image_only` or `multi_modal`.
- `<metadata_setting>`: Select according to the specific attributes you wish to include or exclude in the model training. For all three datasets, options include incorporating all available metadata or focusing on specific aspects such as sex or age(`all`, `sex`, `age`). The PAD dataset offers additional configurations that allow for more detailed exclusions or inclusions, such as skin tone or lifestyle factors like smoking and drinking status(`skin`, `without_smoke`, `without_drink`). For a comprehensive list of all available settings and their specific implications, refer to the `config.json` file.

#### Examples:

**Training a multimodal model on the ODIR dataset with age metadata only:**

```bash
python train.py --dataset_name ODIR --model_str multi_modal --setting age
```

**Training an image-only model on the PAD dataset:**

```bash
python train.py --dataset_name PAD --model_str image_only
```

For a complete list of metadata settings for the PAD dataset, please refer to the `config.json` file provided in the repository.

### 2. Evaluation

You can use the following command. Please remember replace the root path with your own dataset path

To evaluate image-only model:

```bash
python eval.py  --dataset_name <dataset_name> --model_str image_only
```

To evaluate multimodal model:

```bash
python eval.py  --dataset_name <dataset_name> --model_str multi_modal --setting <metadata_setting> --model_file <pretrained_model_file_path> --result_dir <eval_dir>
```

## Large Model-Guided Evaluation of Attributes Relevance

1. #### **Download the Pre-trained Model**:

Obtain the pre-trained weights of the [LLaVA-Med v1.5](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b) model from Hugging Face's model repository. This model is designed to offer advanced capabilities for medical image and text analysis.

2. #### **Running Similarity Evaluations**:

Utilize the following command to compute similarity metrics between medical images and textual descriptions, assessing how well the attributes align with clinical findings and outcomes. 

```
cd similarity_cal
python text_image_similarity.py
python text_text_similarity.py
```

Please remember replace the `image_base_path`, `path_to_text_descriptions` and `model_path` with the actual paths to your datasets. Likewise, `<model_path>` should be replaced with the location where you have stored the downloaded LLaVA-Med v1.5 model weights.

# 
Please contact	**23210240057@m.fudan.edu.cn** if you have questions.
