# The BRIDGE method from the "rLLM: Relational Table Learning with LLMs" paper.
# ArXiv: https://arxiv.org/abs/2407.20157

# Datasets  TLF2K
# Acc       0.471
# MyAcc

# TODO not done yet

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
from rllm.nn.conv.graph_conv import GCNConv
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.models import TableEncoder, GraphEncoder

from models import MultiTableBridge
from rllm_loader import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data, g = load_dataset(dataset_name='tlf2k', load_graph=True)

target_table = data['artists'].to(device)
y = data['artists'].y.long().to(device)

edge_index1 = g['user', 'likes', 'artist'].edge_index
edge_weight1 = g['user', 'likes', 'artist'].listening_cnt.to(torch.float)
edge_index2 = g['user', 'friends_with', 'user'].edge_index
rev_edge_index2 = edge_index2.flip(0)

adj = torch.sparse_coo_tensor(
    indices=adj,
    values=torch.ones(adj.size(1)),
    size=(len(target_table), len(target_table)),
).to(device)
g = GraphData(adj=adj)

####################################################################
paper_emb_size = paper_embeddings.size(1)
table_transform = TabTransformerTransform(
    out_dim=paper_emb_size, metadata=target_table.metadata
)
target_table = table_transform(data=target_table)

author_emb_size = data['author_embeddings'].size(1)
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
# non_table_dict = {
#     'paper': paper_embeddings[len(target_table):, :],
# }
non_table_dict = {
    'paper': paper_embeddings[:, :],
    'author': data['author_embeddings'].to(device),
}
table_embed_dim_dict = {
    'paper': paper_emb_size * 2,
    'author': author_emb_size * 2,
}
node_embed = paper_emb_size * 2
####################################################################

# Split data
train_mask, val_mask, test_mask = (
    target_table.train_mask,
    target_table.val_mask,
    target_table.test_mask,
)

# Set up model and optimizer
t_encoder_paper = TableEncoder(
    in_dim=paper_emb_size,
    out_dim=paper_emb_size,
    table_conv=TabTransformerConv,
    metadata=target_table.metadata,
)
t_encoder_author = TableEncoder(
    in_dim=author_emb_size,
    out_dim=author_emb_size,
    table_conv=TabTransformerConv,
    metadata=data['authors'].metadata,
)
g_encoder = GraphEncoder(
    in_dim=node_embed,
    out_dim=target_table.num_classes,
    graph_conv=GCNConv,
)
model = MultiTableBridge(
    target_table='paper',
    lin_input_dim_dict=table_embed_dim_dict,
    graph_dim=node_embed,
    table_encoder_dict={
        'paper': t_encoder_paper,
        'author': t_encoder_author,
    },
    graph_encoder=g_encoder,
).to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd,
)
####################################################################

def train() -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(
        table_dict=table_dict,
        non_table_dict=non_table_dict,
        edge_index=adj,
    )
    loss = F.cross_entropy(logits[train_mask].squeeze(), y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test():
    model.eval()
    logits = model(
        table_dict=table_dict,
        non_table_dict=non_table_dict,
        edge_index=adj,
    )
    preds = logits.argmax(dim=1)

    accs = []
    for mask in [train_mask, val_mask, test_mask]:
        correct = float(preds[mask].eq(y[mask]).sum().item())
        accs.append(correct / int(mask.sum()))
    return accs


start_time = time.time()
best_val_acc = best_test_acc = 0
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    train_acc, val_acc, test_acc = test()
    print(
        f"Epoch: [{epoch}/{args.epochs}]"
        f"Loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
        f"val_acc: {val_acc:.4f} test_acc: {test_acc:.4f} "
    )
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc

print(f"Total Time: {time.time() - start_time:.4f}s")
print(
    "BRIDGE result: "
    f"Best Val acc: {best_val_acc:.4f}, "
    f"Best Test acc: {best_test_acc:.4f}"
)
