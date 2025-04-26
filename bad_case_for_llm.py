import os
import os.path as osp
import json
import pickle
import torch

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