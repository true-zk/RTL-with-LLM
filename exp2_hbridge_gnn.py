# There may be some bugs.
# Res is extremely bad.

from typing import Dict

import torch
from torch import Tensor

from rllm.data import TableData
from rllm.transforms.table_transforms import TabTransformerTransform

from rllm_loader import load_dataset
from models import HGraphNN
from _utils import print_success


ADD_EMBED = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1 load and prepare data
data_dict, hgraph = load_dataset("tacm12k", load_graph=True)
# del hgraph['author', 'writes', 'paper']
# del hgraph['paper', 'written_by', 'author']
# del hgraph['author']
# print(hgraph)
# print(hgraph.metadata())
# exit()
hgraph = hgraph.to(device)

data_table_dict: Dict[str, TableData] = {
    'paper': data_dict["papers"].to(device),
    'author': data_dict["authors"].to(device),
}

non_table_dict: Dict[str, Tensor] = {
    'paper': data_dict["paper_embeddings"].to(device),
    'author': data_dict["author_embeddings"].to(device),
}

# transform data
table_transform_dict = {
    'paper': TabTransformerTransform(
        out_dim=non_table_dict["paper"].size(1),
        metadata=data_table_dict["paper"].metadata
    ),
    'author': TabTransformerTransform(
        out_dim=non_table_dict["author"].size(1),
        metadata=data_table_dict["author"].metadata
    )
}

data_table_dict = {
    k: table_transform_dict[k](v) for k, v in data_table_dict.items()
}

# data for hgraphNN model
x_dict = {
    k: torch.concat(
        [attr.to(torch.float) for attr in t.feat_dict.values()],
        dim=-1,
    )
    for k, t in data_table_dict.items()
}
if ADD_EMBED:
    x_dict["paper"] = torch.concat([x_dict["paper"], non_table_dict["paper"]], dim=-1)
    x_dict["author"] = torch.concat([x_dict["author"], non_table_dict["author"]], dim=-1)

edge_index_dict = hgraph.edge_index_dict()

# dataset and labels
train_mask, val_mask, test_mask = (
    data_dict["papers"].train_mask,
    data_dict["papers"].val_mask,
    data_dict["papers"].test_mask,
)
y = data_dict["papers"].y.long().to(device)

print_success("Data loaded and transformed successfully.")

# 2 create model
model = HGraphNN(
    target_node_type="paper",
    in_dim={
        k: x.size(1)
        for k, x in x_dict.items()
    },
    hidden_dim=128,
    out_dim=y.unique().size(0),
    metadata=hgraph.metadata(),
    num_heads=4,
    dropout=0.5,
    num_layers=1,
)
# for name, param in model.named_parameters():
#     print(name, param.size())
# exit()
model = model.to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=5e-4,
)

print_success("Model created successfully.")

cnt = 2
# 3 train and test
def train_oncec() -> float:
    model.train()
    optimizer.zero_grad()

    out = model(x_dict, edge_index_dict)
    # print(out[train_mask].size(), y[train_mask].size())
    # exit()
    loss = torch.nn.functional.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test() -> float:
    global cnt
    model.eval()
    out = model(x_dict, edge_index_dict)
    pred = out.argmax(dim=-1)
    print(pred[train_mask].unique())
    print(pred[val_mask].unique())
    print(pred[test_mask].unique())
    accs = [
        pred[train_mask].eq(y[train_mask]).float().mean(),
        pred[val_mask].eq(y[val_mask]).float().mean(),
        pred[test_mask].eq(y[test_mask]).float().mean(),
    ]
    cnt -= 1
    # if cnt < 1:
    #     print(out[:10])
    #     print(pred[:10])
    #     print(pred[100:110])
    #     print(pred[-10:])
    #     print(y[:10])
    #     print(y[100:110])
    #     print(y[-10:])
    #     exit()
    return accs


# print(y.unique().size(0))


for epoch in range(1, 100):
    before = model.readout.weight.clone().detach()
    loss = train_oncec()
    after = model.readout.weight.clone().detach()
    # print(torch.equal(before, after))  # 如果为 False，说明更新了
    # print(torch.norm(after - before))  # 可以量化变化

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         if param.grad is not None:
    #             print(name, param.grad.norm())
    #         else:
    #             print(name, "None")
    # exit()
    accs = test()
    print(
        f"Epoch {epoch:03d} | "
        f"Loss {loss:.4f} | "
        f"Train {accs[0]:.4f} | "
        f"Val {accs[1]:.4f} | "
        f"Test {accs[2]:.4f}"
    )

print_success("Training and testing completed successfully.")
