import os
import os.path as osp
import json
import pickle
import torch
import numpy as np

dataset = "tacm12k"
y_path = osp.join('./tag_data', dataset, 'y.pt')
with open(y_path, 'rb') as f:
    y = torch.load(f)
print(len(y))

y = y.tolist()

preds_path = osp.join('./prompt_2', dataset, 'result.json')
with open(preds_path, 'r') as f:
    preds = json.load(f)

################################################################
# bad_case = []
# bad_case_5 = []

from _utils import llm_preds_2_enhence_vec

# mapping = label2y_map_dict[dataset]


llm_preds_list = [i['Answer'] for i in preds]

llm_enhance_vec = llm_preds_2_enhence_vec(
    llm_preds_l=llm_preds_list,
    dataset_name='tacm12k',
    repeat_l=[1, 1, 1, 1, 1]
)

print(llm_preds_list[:10])
print(llm_enhance_vec[:10])
print(y[:10])

###########################################
with open(os.path.join('./prompt_2', dataset, '1.json'), 'r') as f:
    prompts = json.load(f)
    print(prompts[6])
    print(prompts[9])


############################################################
llm_preds = llm_enhance_vec[:, 0]
print(llm_preds[:10])
print(y[:10])
y = torch.tensor(y)
true_mask_llm = llm_preds == y
true_mask_llm = true_mask_llm.to(torch.long).tolist()

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 5))
plt.bar(range(len(true_mask_llm)), true_mask_llm, color='skyblue')
plt.title('LLM Correct Predictions')
plt.xlabel('Index')
plt.ylabel('Prediction')
# plt.legend()
# plt.show()
plt.savefig('./llm_true_and_false.png')


with open('10_preds_mbridge.pt', 'rb') as f:
    preds_10_times = torch.load(f)


correct_cnts = torch.zeros_like(y, dtype=torch.int)
for preds in preds_10_times:
    true_mask = preds.cpu() == y
    correct_cnts += true_mask

correct_cnts = correct_cnts.tolist()

fig = plt.figure(figsize=(10, 5))
# plt.plot(correct_cnts, [0] * len(correct_cnts), 'ro', label='False')
plt.bar(range(len(correct_cnts)), correct_cnts, color='skyblue')
plt.title('MBRIDGE Correct Predictions Count')
plt.xlabel('Index')
plt.ylabel('Prediction')
# plt.show()
plt.savefig('./mbridge_correct_preds.png')


fig = plt.figure(figsize=(10, 5))
plt.bar(range(len(correct_cnts)), correct_cnts, color='skyblue')
plt.bar(range(len(correct_cnts)), true_mask_llm, color='red')
plt.title('Correct Predictions')
plt.xlabel('Index')
plt.ylabel('Prediction')
# plt.legend()
# plt.show()
plt.savefig('./llm_and_mbridge_correct_preds.png')
