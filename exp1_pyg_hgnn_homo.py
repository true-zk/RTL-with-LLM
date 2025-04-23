# There may be some bugs.
# Res is extremely bad.

from typing import Tuple
import argparse

import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData

from models import PygGCN
from pyg_loader import build_pyg_hdata_from_sjtutables


# Global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Functions
def get_data_model(
    dataset_name: str = "tacm12k",
    with_embed: bool = False,
) -> Tuple[torch.nn.Module, HeteroData]:
    r"""
    Get data and model for training.
    Args:
        dataset_name (str): Name of the dataset.
            ['tacm12k', 'tml1m', 'tlf2k']
        with_embed (bool): Whether to use embeddings.
    """
    data = build_pyg_hdata_from_sjtutables(dataset_name, with_embed=with_embed)
    y = data[data.target_node_type].y
    clses = len(data[data.target_node_type].y.unique())
    train_mask = data[data.target_node_type].train_mask
    val_mask = data[data.target_node_type].val_mask
    test_mask = data[data.target_node_type].test_mask

    x = data[data.target_node_type].x
    edge_index = data['paper', 'cites', 'paper'].edge_index

    model = PygGCN(
        in_dim=x.size(1),
        hidden_dim=128,
        out_dim=clses,
        num_layers=2,
    )

    return model.to(device), x.to(device) , edge_index.to(device), y.to(device), (train_mask, val_mask, test_mask)


def train_once(model, x, edge_index, y, masks, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    train_mask = masks[0]
    # print(y[:10])
    # print(out[:10])
    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_once(model, x, edge_index, y, masks, epoch):
    model.eval()
    preds = model(x, edge_index).argmax(dim=-1)
    accs = []
    for set_ in masks:
        acc = (preds[set_] == y[set_]).sum() / set_.sum()
        accs.append(acc.item())

    return accs


def train_and_test(model, x, edge_index, y, masks, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    for epoch in range(epochs):
        # if epoch == 2:
        #     exit()
        loss = train_once(model, x, edge_index, y, masks, optimizer)
        accs = test_once(model, x, edge_index, y, masks, epoch)
        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Loss: {loss:.4f}, "
            f"Train Acc: {accs[0]:.4f}, "
            f"Val Acc: {accs[1]:.4f}, "
            f"Test Acc: {accs[2]:.4f}."
        )


model, x, edge_index, y, masks = get_data_model(
    dataset_name="tacm12k",
    with_embed=False,
)
# print(data.x_dict)
with torch.no_grad():  # Initialize lazy modules.
    out = model(x, edge_index)
train_and_test(model, x, edge_index, y, masks, epochs=200)
