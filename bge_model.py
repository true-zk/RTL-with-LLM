from transformers import AutoTokenizer, AutoModel

from config import (
    EMBEDDING_MODEL,
)

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL)

# import torch
# texts = ["How are you?", "What is your name?"]

# inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
# print(inputs)
# outputs = model(**inputs)

# embeddings = outputs.last_hidden_state[:, 0, :]  # B, N, D

# embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
# print(embeddings)
