import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib.cm as cm
import random
from matplotlib.lines import Line2D
import torch.nn.functional as F
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["baseline", "cl", "stage1"])
args = parser.parse_args()

MODEL_PATHS = {
    "baseline": "bert-base-multilingual-cased",
    "stage1": "paulbontempo/bert-tagalog-mlm-stage1",
    "cl": "paulbontempo/bert-tagalog-dependency-cl"
}

MODEL_NAMES = {
    "baseline": "Baseline",
    "stage1": "Finetune",
    "cl": "CL"
}

model_name = MODEL_PATHS[args.model]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

sample_size = 2000
all_triples = []
labels = []

with open("checked_graphs.jsonl", "r", encoding="utf-8") as file: #loading triples from graph data
    for line in file:
        data = json.loads(line)
        for key, lists in data.items():
            for triple_list in lists:
                for item in triple_list:
                    tr = tuple(item["triple"])
                    all_triples.append(tr)
                    labels.append(0 if item.get("neg_sample") else 1)

positive_triples = [tr for i, tr in enumerate(all_triples) if labels[i] == 1] #positive triples for accurate tSNE representation
negative_triples = [tr for i, tr in enumerate(all_triples) if labels[i] == 0]

sampled_triples = positive_triples
head_texts = [tr[0] for tr in sampled_triples]
dep_texts  = [tr[1] for tr in sampled_triples]
relations  = [tr[2] for tr in sampled_triples]

#get tokens from the model
head_tokens = tokenizer(head_texts, return_tensors="pt", padding=True, truncation=True)
dep_tokens  = tokenizer(dep_texts, return_tensors="pt", padding=True, truncation=True)

def mean_pooling(tokens, model): #for handling multiple tokens per input
    outputs = model(**tokens).last_hidden_state
    attention_mask = tokens['attention_mask'].unsqueeze(-1)
    masked = outputs * attention_mask
    summed = masked.sum(dim=1)
    counts = attention_mask.sum(dim=1)
    return summed / counts

    
# W/ Contrastive 
with torch.no_grad():
    head_embs = mean_pooling(head_tokens, model)
    dep_embs  = mean_pooling(dep_tokens, model)
    embeddings = torch.cat([head_embs, dep_embs], dim=-1)
    embeddings = F.normalize(embeddings, p=2, dim=1)       # row-wise L2 normalization
    embeddings = embeddings.cpu().numpy()
#/
    

from sklearn.decomposition import PCA
embeddings_50d = PCA(n_components=30).fit_transform(embeddings)

# perform tSNE 
pos_2d = TSNE(init='pca', learning_rate=100, metric='cosine', n_components=2, perplexity=30, random_state=42, max_iter=2000).fit_transform(embeddings_50d)

# consistent color labels for each relation type 
unique_relations = sorted(set(relations))  # sort alphabetically
cmap = cm.get_cmap('tab20', len(unique_relations))
relation_to_color = {rel: cmap(i) for i, rel in enumerate(unique_relations)}

plt.figure(figsize=(8,6))

legend_handles = [
    Line2D([0], [0], marker='o', color='w', label=rel,
           markerfacecolor=color, markersize=6)
    for rel, color in relation_to_color.items()
]
plt.legend(handles=legend_handles, title="Relations", bbox_to_anchor=(1.05, 1), loc='upper left')

for i, (x, y) in enumerate(pos_2d):
    color = relation_to_color[relations[i]]
    plt.scatter(x, y, color=color, s=10)

plt.title(f"t-SNE of Triples ({MODEL_NAMES[args.model]})")
plt.xlabel("TSNE-1")
plt.ylabel("TSNE-2")
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig(f"tSNE_Triples_{MODEL_NAMES[args.model]}.png")
# plt.show()
