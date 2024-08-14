import sys
# import cosine
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import os

sys.path.append('/similarity_cal/llavamed')
from llavamed.llava.model.builder import load_pretrained_model
from llavamed.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

model_path = "similarity_cal/llava-med-v1.5-mistral-7b"
image_base_path = 'Dataset/PADdata/images/'
path_to_text_descriptions = 'similarity_cal/updated_metadata.csv'
results_path = "similarity_results"

data = pd.read_csv(path_to_text_descriptions)
model_name = "LlavaMistralForCausalLM"
device = "cuda"
load_8bit = False
load_4bit = False

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name, load_8bit, load_4bit, device=device)

print('loaded')


results = {}
text_columns = data.columns.drop('img_id')
similarity_matrix = []
results_every = pd.DataFrame(columns=['img_id'] + list(text_columns))
i = 0

for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing images"):
    i += 1
    texts = [str(row[col]) for col in text_columns]
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded_inputs['input_ids'].to(device)
    with torch.no_grad():
        model_output = model(input_ids=input_ids)
    text_features = model_output['text_features']
    text_features = text_features.mean(dim=1)
    print('text_features:', text_features.shape)
    similarities = cosine_similarity(text_features.cpu().numpy())

    print('similarities:', similarities.shape)
    similarity_matrix.append(similarities)
    for j, col in enumerate(text_columns):
        new_row = pd.DataFrame([{'img_id': row['img_id'], **dict(zip(text_columns, similarities))}])
        results_every = pd.concat([results_every, new_row], ignore_index=True)

similarity_matrix = np.array(similarity_matrix)
print('similarity_matrix:', similarity_matrix.shape)
avg_similarities = similarity_matrix.mean(axis=0)
print('avg_similarities:', avg_similarities.shape)
results_df = pd.DataFrame(avg_similarities, index=text_columns, columns=text_columns)
results_df.to_csv(os.path.join(results_path, 'text_similarity_matrix.csv'), index=True)
results_every.to_csv(os.path.join(results_path, 'text_similarity_every.csv'), index=False)
print('done')