# There may be some bugs.
# Res is extremely bad.

from typing import Dict

import torch
from torch import Tensor

from rllm.data import TableData
from rllm.transforms.table_transforms import TabTransformerTransform

from rllm_loader import load_dataset
from models import TrivialGNN
from _utils import print_success


ADD_EMBED = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1 load and prepare data
data_dict, _ = load_dataset("tacm12k", load_graph=True)

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

del x_dict["author"]

x = x_dict["paper"]

edge_list = torch.tensor(
    data_dict['citations'].df[["paper_id", "paper_id_cited"]].values,
    dtype=torch.long,
).T
edge_index = torch.concat(
    [
        edge_list,
        edge_list.flip(0),
    ],
    dim=-1,
)
edge_index = torch.sparse_coo_tensor(
    indices=edge_index,
    values=torch.ones(edge_index.size(1), dtype=torch.float),
    size=(x.size(0), x.size(0)),
)
edge_index = edge_index.to(device)


# dataset and labels
train_mask, val_mask, test_mask = (
    data_dict["papers"].train_mask,
    data_dict["papers"].val_mask,
    data_dict["papers"].test_mask,
)
y = data_dict["papers"].y.long().to(device)

print_success("Data loaded and transformed successfully.")

model = TrivialGNN(
    in_channels=x.size(1),
    hidden_channels=128,
    out_channels=y.unique().size(0),
    num_layers=4,
)
model = model.to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=5e-4,
)

print_success("Model created successfully.")

# 3 train and test
def train_oncec() -> float:
    model.train()
    optimizer.zero_grad()

    out = model(x, edge_index)
    loss = torch.nn.functional.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test() -> float:
    model.eval()
    out = model(x, edge_index)
    pred = out.argmax(dim=-1)
    print(pred[train_mask].unique())
    print(pred[val_mask].unique())
    print(pred[test_mask].unique())
    accs = [
        pred[train_mask].eq(y[train_mask]).float().mean(),
        pred[val_mask].eq(y[val_mask]).float().mean(),
        pred[test_mask].eq(y[test_mask]).float().mean(),
    ]
    return accs


for epoch in range(1, 100):
    loss = train_oncec()
    accs = test()
    print(
        f"Epoch {epoch:03d} | "
        f"Loss {loss:.4f} | "
        f"Train {accs[0]:.4f} | "
        f"Val {accs[1]:.4f} | "
        f"Test {accs[2]:.4f}"
    )

print_success("Training and testing completed successfully.")
