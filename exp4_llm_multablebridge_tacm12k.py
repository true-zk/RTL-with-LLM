# The BRIDGE method from the "rLLM: Relational Table Learning with LLMs" paper.
# ArXiv: https://arxiv.org/abs/2407.20157

# Datasets  TACM12K
# Acc       0.293
# Fuse1    ~0.616
# Fuse2    ~0.645

import time
import argparse
import sys

import torch
import torch.nn.functional as F

sys.path.append("./")
sys.path.append("../")
from rllm.data import GraphData
from rllm.transforms.graph_transforms import GCNTransform
from rllm.transforms.table_transforms import TabTransformerTransform
from rllm.nn.conv.graph_conv import GCNConv, GATConv
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.models import TableEncoder, GraphEncoder

from models import MultiTableBridge
from rllm_loader import load_dataset
from _utils import print_success, print_warning

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data, g = load_dataset(dataset_name='tacm12k', load_graph=True)
paper_embeddings = data['paper_embeddings']
papers_table = data['papers']


target_table = papers_table.to(device)
y = papers_table.y.long().to(device)

paper_embeddings = paper_embeddings.to(device)

edge_index1 = g['paper', 'cites', 'paper'].edge_index
edge_index2 = g['paper', 'written_by', 'author'].edge_index
# paper | author
edge_index2[1] += len(papers_table)
adj = torch.concat(
    [
        edge_index1,
        edge_index2,
        edge_index2.flip(0)
    ],
    dim=-1
)
num_nodes = len(papers_table) + len(data['authors'])
adj = torch.sparse_coo_tensor(
    indices=adj,
    values=torch.ones(adj.size(1)),
    size=(num_nodes, num_nodes),
).to(device)
g = GraphData(adj=adj)

####################################################################
paper_nontable_size = paper_embeddings.size(1)
paper_emb_size = 16  # paper_emb_size = paper_embeddings.size(1) // 2

table_transform = TabTransformerTransform(
    out_dim=paper_emb_size, metadata=target_table.metadata
)
target_table = table_transform(data=target_table)

author_nontable_size = data['author_embeddings'].size(1)
author_emb_size = 16  # author_emb_size = data['author_embeddings'].size(1) // 2
author_table_transform = TabTransformerTransform(
    out_dim=author_emb_size, metadata=data['authors'].metadata
)
data['authors'] = author_table_transform(data=data['authors'])

graph_transform = GCNTransform()
adj = graph_transform(data=g).adj
adj = adj.to(device)

####################################################################
table_dict = {
    'paper': target_table.to(device),
    'author': data['authors'].to(device),
}
non_table_dict = {
    'paper': paper_embeddings[:, :],
    'author': data['author_embeddings'].to(device),
}
####################################################################
# load llm enhencement
import os.path as osp
import json

from _utils import llm_preds_2_enhence_vec


file_path = osp.join(osp.dirname(osp.realpath(__file__)), "prompt_2", "tacm12k")
file_path = osp.join(file_path, "result.json")
with open(file_path, "r") as f:
    llm_enhancement = json.load(f)
    llm_preds_list = [i['Answer'] for i in llm_enhancement]

llm_enhance_vec = llm_preds_2_enhence_vec(
    llm_preds_l=llm_preds_list,
    dataset_name='tacm12k',
    repeat_l=[1, 1, 1, 1, 1]
)

llm_enhance_vec = llm_enhance_vec.to(torch.long).to(device)  # n * 5
######
keep_mask = torch.rand_like(y, dtype=torch.float) <= 0.5888
flip_indices = (~keep_mask).nonzero(as_tuple=False).squeeze()
y_noise = y.clone()
for idx in flip_indices:
    correct_label = y[idx]
    choices = torch.tensor([i for i in range(13) if i != correct_label])
    new_label = choices[torch.randint(len(choices), (1,))]
    y_noise[idx] = new_label

print("acc of simulation of llm pred:", (y == y_noise).float().mean())

llm_enhance_vec = y_noise.unsqueeze(-1).repeat_interleave(5, dim=1)
llm_enhance_vec[:, 1:] = torch.randint(0, 13, (len(y), 4))

llm_vec_size = llm_enhance_vec.size(1)  # 5
_embed_dim = 16

from models import ImportanceEncoder
llm_vec_encoder = ImportanceEncoder(
    num_labels=target_table.num_classes + 1,  # for [UNK]
    embed_dim=_embed_dim,
    input_size=llm_vec_size,
    weight=[2.0, 0.3, 0.1, 0.1, 0.1],
)

llm_embed_size = llm_vec_size * _embed_dim
####################################################################

# Split data
train_mask, val_mask, test_mask = (
    target_table.train_mask,
    target_table.val_mask,
    target_table.test_mask,
)

####################################################################
# Set up model and optimizer
# table encoder
t_encoder_paper = TableEncoder(  # paper_encoder_out = paper_emb_size
    in_dim=paper_emb_size,
    out_dim=paper_emb_size,
    table_conv=TabTransformerConv,
    metadata=target_table.metadata,
)
t_encoder_author = TableEncoder(  # author_encoder_out = author_emb_size
    in_dim=author_emb_size,
    out_dim=author_emb_size,
    table_conv=TabTransformerConv,
    metadata=data['authors'].metadata,
)

# linear before graph conv
lin_input_dim_dict = {
    'paper': paper_emb_size + paper_nontable_size + llm_embed_size,  # paper_encoder_out + non_table + llm
    'author': author_emb_size + author_nontable_size,  # author_encoder_out + non_table
}

lin_out_dim = graph_in_dim = 64
graph_out_dim = target_table.num_classes

g_encoder = GraphEncoder(
    in_dim=graph_in_dim,
    out_dim=graph_out_dim,
    graph_conv=GCNConv,
)

# tout_1 + non_table + llm | tout_2 + non_table
model1 = MultiTableBridge(
    target_table='paper',
    # dim
    lin_input_dim_dict=lin_input_dim_dict,
    graph_dim=graph_in_dim,
    # encoder
    table_encoder_dict={
        'paper': t_encoder_paper,
        'author': t_encoder_author,
    },
    graph_encoder=g_encoder,
    dropout=0.6,
    # for llm enhancement
    llm_vec_encoder=llm_vec_encoder,
)
optimizer1 = torch.optim.AdamW(
    model1.parameters(),
    lr=args.lr,
    # weight_decay=args.wd,
)

model_optim_dict = {
    'MultiTableBRIDGE': (model1, optimizer1),
}


from models import FuseII
from copy import deepcopy


lin_input_dim_dict['paper'] = paper_emb_size + paper_nontable_size
graph_in_dim = 64
graph_out_dim = 32

g_encoder2 = GraphEncoder(
    in_dim=graph_in_dim,
    out_dim=graph_out_dim,
    graph_conv=GCNConv,
)

llm_vec_encoder2 = ImportanceEncoder(
    num_labels=target_table.num_classes + 1,  # for [UNK]
    embed_dim=8,
    input_size=llm_vec_size,
    weight=[1.0, 0.3, 0.1, 0.1, 0.1],
)

model2 = FuseII(
    target_table='paper',
    lin_input_dim_dict=lin_input_dim_dict,
    graph_dim=graph_in_dim,
    table_encoder_dict={
        'paper': deepcopy(t_encoder_paper),
        'author': deepcopy(t_encoder_author),
    },
    graph_encoder=g_encoder2,
    dropout=0.6,
    bridge_out_dim=graph_out_dim,
    llm_encoder_out_dim=8 * llm_vec_size,
    hidden_dim=32,
    output_dim=target_table.num_classes,
    llm_vec_encoder=llm_vec_encoder2,
)

optimizer2 = torch.optim.AdamW(
    model2.parameters(),
    lr=args.lr,
    # weight_decay=args.wd,
)

model_optim_dict['FuseII'] = (model2, optimizer2)
# del model_optim_dict['MultiTableBRIDGE']

print_success("all set up")
print(f"models: {model_optim_dict.keys()}")

####################################################################

####################################################################

def train(model, optimizer) -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(
        table_dict=table_dict,
        non_table_dict=non_table_dict,
        edge_index=adj,
        llm_vec=llm_enhance_vec,
    )
    loss = F.cross_entropy(logits[train_mask].squeeze(), y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model):
    model.eval()
    logits = model(
        table_dict=table_dict,
        non_table_dict=non_table_dict,
        edge_index=adj,
        llm_vec=llm_enhance_vec,
    )
    preds = logits.argmax(dim=1)

    accs = []
    for mask in [train_mask, val_mask, test_mask]:
        correct = float(preds[mask].eq(y[mask]).sum().item())
        accs.append(correct / int(mask.sum()))
    return accs


def train_one_model(model_name, model, optim):
    start_time = time.time()
    best_val_acc = best_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, optim)
        train_acc, val_acc, test_acc = test(model)
        print(
            f"Epoch: [{epoch}/{args.epochs}]"
            f"Loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
            f"val_acc: {val_acc:.4f} test_acc: {test_acc:.4f} "
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

    print_success(f"Total Time: {time.time() - start_time:.4f}s")
    print_success(
        f"{model_name} result: "
        f"Best Val acc: {best_val_acc:.4f}, "
        f"Best Test acc: {best_test_acc:.4f}"
    )
    return best_val_acc, best_test_acc


res_d = {}
for model_name, (model, optim) in model_optim_dict.items():
    # if model_name == "MultiTableBRIDGE":
    #     continue
    res_d[model_name] = {}
    best_val_acc, best_test_acc = train_one_model(model_name, model.to(device), optim)
    res_d[model_name]['val'] = best_val_acc
    res_d[model_name]['test'] = best_test_acc
    if model_name == "FuseII":
        print_warning(str(model.p))


print_success("all done")
for model_name, res in res_d.items():
    print(f"{model_name}: {res['val']:.4f}, {res['test']:.4f}")
