# The BRIDGE method from the "rLLM: Relational Table Learning with LLMs" paper.
# ArXiv: https://arxiv.org/abs/2407.20157

# Datasets  TML1M
# Acc       0.397
# Fuse      0.343

import time
import argparse
import sys
import os.path as osp

import torch
import torch.nn.functional as F

sys.path.append("./")
from rllm.datasets import TML1MDataset
from rllm.transforms.graph_transforms import GCNTransform
from rllm.transforms.table_transforms import TabTransformerTransform
from rllm.nn.conv.graph_conv import GCNConv
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.models import TableEncoder, GraphEncoder, BRIDGE
from bridge.utils import build_homo_graph, reorder_ids

from models import LLMBRIDGE

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
path = osp.join(osp.dirname(osp.realpath(__file__)), "./", "data")
dataset = TML1MDataset(cached_dir=path, force_reload=True)

# Get the required data
(
    user_table,
    _,
    rating_table,
    movie_embeddings,
) = dataset.data_list
emb_size = movie_embeddings.size(1)
user_size = len(user_table)

ordered_rating = reorder_ids(
    relation_df=rating_table.df,
    src_col_name="UserID",
    tgt_col_name="MovieID",
    n_src=user_size,
)
target_table = user_table.to(device)
y = user_table.y.long().to(device)
movie_embeddings = movie_embeddings.to(device)

# Build graph
graph = build_homo_graph(
    relation_df=ordered_rating,
    n_all=user_size + movie_embeddings.size(0),
).to(device)

# Transform data
table_transform = TabTransformerTransform(
    out_dim=emb_size, metadata=target_table.metadata
)
target_table = table_transform(target_table)
graph_transform = GCNTransform()
adj = graph_transform(graph).adj

# Split data
train_mask, val_mask, test_mask = (
    user_table.train_mask,
    user_table.val_mask,
    user_table.test_mask,
)

###########################################################
keep_mask = torch.rand_like(y, dtype=torch.float) <= 0.3038
flip_indices = (~keep_mask).nonzero(as_tuple=False).squeeze()
y_noise = y.clone()
for idx in flip_indices:
    correct_label = y[idx]
    choices = torch.tensor([i for i in range(target_table.num_classes) if i != correct_label])
    new_label = choices[torch.randint(len(choices), (1,))]
    y_noise[idx] = new_label

print("acc of simulation of llm pred:", (y == y_noise).float().mean())

llm_enhance_vec = y_noise.unsqueeze(-1).repeat_interleave(5, dim=1)
llm_enhance_vec[:, 1:] = torch.randint(0, target_table.num_classes, (len(y), 4))
llm_enhance_vec = llm_enhance_vec.to(device)

llm_vec_size = llm_enhance_vec.size(1)  # 5
_embed_dim = 8

from models import ImportanceEncoder
llm_vec_encoder = ImportanceEncoder(
    num_labels=target_table.num_classes + 1,  # for [UNK]
    embed_dim=_embed_dim,
    input_size=llm_vec_size,
    weight=[1.0, 0.1, 0.1, 0.1, 0.1],
)

llm_embed_size = llm_vec_size * _embed_dim

###########################################################
bridge_out_dim = 32
# Set up model and optimizer
t_encoder = TableEncoder(
    in_dim=emb_size,
    out_dim=emb_size,
    table_conv=TabTransformerConv,
    metadata=target_table.metadata,
)
g_encoder = GraphEncoder(
    in_dim=emb_size,
    out_dim=bridge_out_dim,
    graph_conv=GCNConv,
)

bridge = BRIDGE(
    table_encoder=t_encoder,
    graph_encoder=g_encoder,
).to(device)

model = LLMBRIDGE(
    bridge=bridge,
    bridge_out_dim=bridge_out_dim,
    llm_encoder_out_dim=llm_embed_size,
    hidden_dim=16,
    output_dim=target_table.num_classes,
    llm_vec_encoder=llm_vec_encoder,
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd,
)


def train() -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(
        table=user_table,
        non_table=movie_embeddings,
        adj=adj,
        llm_vec=llm_enhance_vec,
    )

    loss = F.cross_entropy(logits[train_mask].squeeze(), y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test():
    model.eval()
    logits = model(
        table=user_table,
        non_table=movie_embeddings,
        adj=adj,
        llm_vec=llm_enhance_vec,
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

print(model.p)
