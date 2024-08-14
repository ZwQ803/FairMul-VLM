import sys
# import cosine
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
sys.path.append('/mnc/zwq/knowledge/LLaVA-Med-main')  # 修改为包含 llava 包的目录

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

model_path = "/mnc/zwq/knowledge/llava-med-v1.5-mistral-7b"  # 模型文件目录
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # 模型名称
model_name = "LlavaMistralForCausalLM"
device = "cuda"  # 设备选择, 例如 'cuda' 或 'cpu'
load_8bit = False  # 根据需求设置
load_4bit = False  # 根据需求设置

# 调用函数加载模型
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name, load_8bit, load_4bit, device=device)

print('loaded')

from PIL import Image
import torch

# 假设你已经有了一些文本和对应的图像文件路径
text = "smoke."
image_path = "/mnc/zwq/Dataset/PADdata/images/PAT_1000_31_620.png"

# 处理文本
encoded_input = tokenizer(text, return_tensors="pt")
input_ids = encoded_input['input_ids'].to(device)

# 处理图像
image = Image.open(image_path).convert("RGB")
# image = image.resize((224, 224))
# processed_image = image_processor(image)  # 假设这是预处理函数
# image_tensor = processed_image.unsqueeze(0).to(device)  # 添加batch维度
image_tensor = process_images([image], image_processor, model.config)[0]
image_tensor=image_tensor.unsqueeze(0).half().cuda()
# 调用模型获取特征
with torch.no_grad():
    model_output = model(input_ids=input_ids, images=image_tensor)

# 获取文本和图像特征
text_features = model_output['text_features']
image_features = model_output['image_features']

# 可以根据需要进一步处理或保存这些特征
print("Text features shape:", text_features.shape)
print("Image features shape:", image_features.shape)

image_features_mean = image_features.mean(dim=1).squeeze()  # 减少到 [1, 4096] 然后 squeeze 到 [4096]
text_features_mean = text_features.mean(dim=1).squeeze()    # 同上

# 将特征转移到CPU并转换为NumPy数组
image_features_np = image_features_mean.cpu().numpy()
text_features_np = text_features_mean.cpu().numpy()

# 计算余弦相似度
similarity_1 = 1 - cosine(image_features_np, text_features_np)
print("Cosine similarity: ", similarity_1)
#
# image_features = image_features.cpu()  # Move to CPU
# text_features = text_features.cpu()    # Move to CPU
#
# # Convert PyTorch tensors to NumPy arrays
# image_features_np = image_features.numpy()
# text_features_np = text_features.numpy()
#
# image_features_mean = image_features_np.mean(axis=0)
# text_features_mean = text_features_np.mean(axis=0)
#
# # 计算余弦相似度
# similarity_1 = 1 - cosine(image_features_mean, text_features_mean)
# print("Cosine similarity: ", similarity_1)
# #
# similarity_1 = 1 - cosine(image_features, text_features)
# print('1:',similarity_1)
# similarity_2 = cosine_similarity(image_features.numpy(), text_features.numpy())
#
# print('2:',similarity_2)


#
# csv_path = '/mnc/zwq/knowledge/LLaVA-Med-main/metadata.csv'
# data = pd.read_csv(csv_path)
# image_folder = '/mnc/zwq/Dataset/PADdata/images/'
#
# # image_file = line["image"]
# # image = Image.open(os.path.join(image_folder, image_file))
# # image_tensor = process_images([image], image_processor, model.config)[0]
#
#
# # 假设你已经有以下函数来处理图像和文本并获取特征
# def get_image_features(image_file):
#     # 加载图像，预处理，并提取特征
#     image = Image.open(os.path.join(image_folder, image_file))
#     image_tensor = process_images([image], image_processor, model.config)[0]
#     #
#     #
#     # image = load_and_preprocess_image(image_path)
#     # image_features = image_processor(image)  # 假设这是你的图像处理函数
#     return image_tensor
#
# def get_text_features(text):
#     # 使用tokenizer和模型提取文本特征
#     inputs = tokenizer(text, return_tensors="pt")
#     outputs = model(**inputs)
#     text_features = outputs.last_hidden_state.mean(dim=1)  # 取平均得到特征
#     return text_features
#
# # 对于每一行数据，计算文本和图像特征的相似度
# similarities = []
# for index, row in data.iterrows():
#     image_features = get_image_features(row['img_id'])
#     text_features = get_text_features(row['diagnostic'])  # 示例：使用diagnostic字段
#     similarity = 1 - cosine(image_features, text_features)
#     similarities.append(similarity)
#
# # 将相似度添加到数据框中
# data['similarity'] = similarities