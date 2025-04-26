# The BRIDGE method from the "rLLM: Relational Table Learning with LLMs" paper.
# ArXiv: https://arxiv.org/abs/2407.20157

# Datasets  TML1M
# Acc       0.397
# MyAcc

# TODO bug fix

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

data, g = load_dataset(dataset_name='tml1m', load_graph=True)

target_table = data['users'].to(device)
y = data['users'].y.long().to(device)

edge_index1 = g['user', 'rates', 'movie'].edge_index
rev_edge_index1 = edge_index1.flip(0)
adj = torch.concat(
    [
        edge_index1,
        rev_edge_index1,
    ],
    dim=-1
)

num_nodes = len(data['users']) + len(data['movies'])

adj = torch.sparse_coo_tensor(
    indices=adj,
    values=torch.ones(adj.size(1)),
    size=(num_nodes, num_nodes),
).to(device)
g = GraphData(adj=adj)

####################################################################
user_embed_size = 256
table_transform = TabTransformerTransform(
    out_dim=user_embed_size, metadata=target_table.metadata
)
target_table = table_transform(data=target_table)

movie_emb_size = data['movie_embeddings'].size(1)
movie_table_transform = TabTransformerTransform(
    out_dim=movie_emb_size, metadata=data['movies'].metadata
)
data['movies'] = movie_table_transform(data=data['movies'])

graph_transform = GCNTransform()
adj = graph_transform(data=g).adj
adj = adj.to(device)

# print(data['movies'].feat_dict)
####################################################################
from rllm.types import ColType
# print(data['movies'].feat_dict[])
# exit()
data['movies'].feat_dict[ColType.NUMERICAL] = torch.ones(
    (len(data['movies']), 64), device=device
)
####################################################################
table_dict = {
    'user': target_table.to(device),
    'movie': data['movies'].to(device),
}

non_table_dict = {
    'user': torch.empty(0, device=device),
    'movie': data['movie_embeddings'].to(device),
}
table_embed_dim_dict = {
    'user': user_embed_size,
    'movie': movie_emb_size + 64,
}
node_embed = user_embed_size * 2

# print(table_embed_dim_dict)
# exit()
####################################################################

# Split data
train_mask, val_mask, test_mask = (
    target_table.train_mask,
    target_table.val_mask,
    target_table.test_mask,
)

# Set up model and optimizer
t_encoder_user = TableEncoder(
    in_dim=user_embed_size,
    out_dim=user_embed_size,
    table_conv=TabTransformerConv,
    metadata=target_table.metadata,
)
t_encoder_movie = TableEncoder(
    in_dim=64,
    out_dim=64,
    table_conv=TabTransformerConv,
    metadata=data['movies'].metadata,
)
g_encoder = GraphEncoder(
    in_dim=node_embed,
    out_dim=target_table.num_classes,
    graph_conv=GCNConv,
)
model = MultiTableBridge(
    target_table='user',
    lin_input_dim_dict=table_embed_dim_dict,
    graph_dim=node_embed,
    table_encoder_dict={
        'user': t_encoder_user,
        'movie': t_encoder_movie,
    },
    graph_encoder=g_encoder,
    dropout=0.3
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
