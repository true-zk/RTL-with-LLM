# There may be some bugs.
# Res is extremely bad.

from typing import Tuple
import argparse

import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData

from models import HeteroSAGE, HAN
from pyg_loader import build_pyg_hdata_from_sjtutables


# Global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Functions
def get_data_model(
    dataset_name: str = "tacm12k",
    with_embed: bool = False,
    model_name: str = "HeteroSAGE",
) -> Tuple[torch.nn.Module, HeteroData]:
    r"""
    Get data and model for training.
    Args:
        dataset_name (str): Name of the dataset.
            ['tacm12k', 'tml1m', 'tlf2k']
        with_embed (bool): Whether to use embeddings.
    """
    data = build_pyg_hdata_from_sjtutables(dataset_name, with_embed=with_embed)
    clses = len(data[data.target_node_type].y.unique())

    if model_name == "HeteroSAGE":
        model = HeteroSAGE(
            target_node_type=data.target_node_type,
            metadata=data.metadata(),
            inchannels={
                node_type: x.size(1) for node_type, x in data.x_dict.items()
            },
            hidden_channels=64,
            out_channels=clses,
            num_layers=2,
            virtual_node_type='user' if dataset_name == 'tlf2k' else None,
            num_virtual_nodes=data['user'].num_nodes if dataset_name == 'tlf2k' else None,
        )
    elif model_name == "HAN":
        model = HAN(
            metadata=data.metadata(),
            target_node_type=data.target_node_type,
            out_channels=clses,
            hidden_channels=64,
            heads=8,
            virtual_node_type='user' if dataset_name == 'tlf2k' else None,
            num_virtual_nodes=data['user'].num_nodes if dataset_name == 'tlf2k' else None,
        )

    return model.to(device), data.to(device)


def train_once(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    train_mask = data[data.target_node_type].train_mask
    y = data[data.target_node_type].y
    # print(y[:10])
    # print(out[:10])
    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_once(model, data, epoch):
    model.eval()
    preds = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)
    # if epoch == 180:
    #     print(preds[:10])
    #     val = data[data.target_node_type]['val_mask']
    #     exit()
    accs = []
    for set_ in ['train_mask', 'val_mask', 'test_mask']:
        mask = data[data.target_node_type][set_]
        y = data[data.target_node_type].y
        acc = (preds[mask] == y[mask]).sum() / mask.sum()
        accs.append(acc.item())

    return accs


def train_and_test(model, data, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    for epoch in range(epochs):
        # if epoch == 2:
        #     exit()
        loss = train_once(model, data, optimizer)
        accs = test_once(model, data, epoch)
        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Loss: {loss:.4f}, "
            f"Train Acc: {accs[0]:.4f}, "
            f"Val Acc: {accs[1]:.4f}, "
            f"Test Acc: {accs[2]:.4f}."
        )


model, data = get_data_model(
    dataset_name="tacm12k",
    with_embed=False,
    model_name="HeteroSAGE",
)
# print(data.x_dict)
with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)
train_and_test(model, data, epochs=200)
