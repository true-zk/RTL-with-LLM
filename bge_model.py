from transformers import AutoTokenizer, AutoModel

from config import (
    EMBEDDING_MODEL,
)

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL)

# import torch
# # 输入文本
# texts = ["How are you?", "What is your name?"]

# # 编码 + 获取模型输出
# inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
# print(inputs)
# outputs = model(**inputs)

# # 取 [CLS] token 的 embedding
# embeddings = outputs.last_hidden_state[:, 0, :]  # B, N, D

# # 归一化
# embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
# print(embeddings)
