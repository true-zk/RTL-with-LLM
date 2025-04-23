from typing import Dict, Tuple, Union, Optional

import torch
from torch import Tensor
import pandas as pd
from pandas import DataFrame
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import to_undirected

from rllm_loader import load_dataset


def simple_norm(attr: Tensor) -> Tensor:
    """
    Normalize the tensor by its L2 norm.
    """
    attr = attr.to(torch.float)
    norm = attr.norm(p=2, dim=0, keepdim=True)
    return attr.div(norm).clamp(min=1e-5)

# a = torch.tensor([
#     [1, 1, 1],
#     [1, 5, 6],
#     [1, 8, 9],
#     [1, 11, 12]
# ], dtype=torch.float)
# print(simple_norm(a))
# exit()


def load_edge_df(
    df: DataFrame,
    src_index_col: str,
    src_mapping: Optional[Union[int, Dict]],
    dst_index_col: str,
    dst_mapping: Optional[Union[int, Dict]],
    edge_attr_col: str = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Load edge DataFrame and convert to edge index format.
    """
    if src_mapping is not None:
        if isinstance(src_mapping, int):
            src = df[src_index_col] + src_mapping
        else:
            src = [src_mapping[i] for i in df[src_index_col]]
    else:
        src = df[src_index_col]

    if dst_mapping is not None:
        if isinstance(dst_mapping, int):
            dst = df[dst_index_col] + dst_mapping
        else:
            dst = [dst_mapping[i] for i in df[dst_index_col]]
    else:
        dst = df[dst_index_col]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = None
    if edge_attr_col is not None:
        edge_attr = torch.tensor(df[edge_attr_col].values, dtype=torch.float)

    return edge_index, edge_attr


def build_pyg_hdata_from_sjtutables(dataset_name, with_embed: bool = False) -> HeteroData:

    data = HeteroData()
    table_dict = load_dataset(dataset_name)

    if dataset_name == "tacm12k":
        x = [attr.to(torch.float) for attr in table_dict['papers'].feat_dict.values()]
        if with_embed:
            x.append(table_dict['paper_embeddings'])
        data['paper'].x = torch.concat(x, dim=1)
        data['paper'].y = table_dict['papers'].y.to(torch.long)
        data['paper'].train_mask = table_dict['papers'].train_mask
        data['paper'].val_mask = table_dict['papers'].val_mask
        data['paper'].test_mask = table_dict['papers'].test_mask
        data.target_node_type = 'paper'

        x = [attr.to(torch.float) for attr in table_dict['authors'].feat_dict.values()]
        if with_embed:
            x.append(table_dict['author_embeddings'])
        data['author'].x = torch.concat(x, dim=1)

        # print(len(data['paper'].x))
        # print(len(data['author'].x))

        edge_index, _ = load_edge_df(
            df=table_dict['citations'].df,
            src_index_col='paper_id',
            src_mapping=None,
            dst_index_col='paper_id_cited',
            dst_mapping=None,
            edge_attr_col=None,
        )
        data['paper', 'cites', 'paper'].edge_index = edge_index
        # print(edge_index[0].min(), edge_index[0].max())
        # print(edge_index[1].min(), edge_index[1].max())
        # data['paper', 'cited_by', 'paper'].edge_index = edge_index.flip(0)

        edge_index, _ = load_edge_df(
            df=table_dict['writings'].df,
            src_index_col='paper_id',
            src_mapping=None,
            dst_index_col='author_id',
            dst_mapping=None,
            edge_attr_col=None,
        )
        data['paper', 'writed_by', 'author'].edge_index = edge_index
        # print(edge_index[0].min(), edge_index[0].max())
        # print(edge_index[1].min(), edge_index[1].max())
        data = ToUndirected()(data)

    elif dataset_name == "tlf2k":
        x = [attr.to(torch.float) for attr in table_dict['artists'].feat_dict.values()]
        data['artist'].x = torch.concat(x, dim=1)
        data['artist'].y = table_dict['artists'].y.to(torch.long)
        data['artist'].train_mask = table_dict['artists'].train_mask
        data['artist'].val_mask = table_dict['artists'].val_mask
        data['artist'].test_mask = table_dict['artists'].test_mask
        data.target_node_type = 'artist'

        data['user'].num_nodes = table_dict['user_artists'].df['userID'].max()

        edge_index, edge_attr = load_edge_df(
            df=table_dict['user_artists'].df,
            src_index_col='userID',
            src_mapping=-1,
            dst_index_col='artistID',
            dst_mapping=-1,
            edge_attr_col='weight',
        )
        data['user', 'likes', 'artist'].edge_index = edge_index
        data['user', 'likes', 'artist'].weight = edge_attr

        edge_index, _ = load_edge_df(
            df=table_dict['user_friends'].df,
            src_index_col='userID',
            src_mapping=-1,
            dst_index_col='friendID',
            dst_mapping=-1,
            edge_attr_col=None,
        )
        edge_index = to_undirected(edge_index)
        data['user', 'friend_with', 'user'].edge_index = edge_index

    elif dataset_name == 'tml1m':
        x = [simple_norm(attr) for attr in table_dict['users'].feat_dict.values()]
        data['user'].x = torch.concat(x, dim=1)
        data['user'].y = table_dict['users'].y.to(torch.long)
        data['user'].train_mask = table_dict['users'].train_mask
        data['user'].val_mask = table_dict['users'].val_mask
        data['user'].test_mask = table_dict['users'].test_mask
        data.target_node_type = 'user'

        x = [simple_norm(attr) for attr in table_dict['movies'].feat_dict.values()]
        if with_embed:
            x.append(table_dict['movie_embeddings'])
        data['movie'].x = torch.concat(x, dim=1)

        edge_index, edge_attr = load_edge_df(
            df=table_dict['ratings'].df,
            src_index_col='UserID',
            src_mapping=-1,
            dst_index_col='MovieID',
            dst_mapping=-1,
            edge_attr_col='Rating',
        )
        data['user', 'rating', 'movie'].edge_index = edge_index
        data['user', 'rating', 'movie'].edge_attr = edge_attr
        data = ToUndirected()(data)

    return data


a = build_pyg_hdata_from_sjtutables('tacm12k', False)
print(a)
print(a.metadata())
# print(a.target_node_type)
# print(a['user'].x[:10])
