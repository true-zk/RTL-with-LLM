import os
from typing import Optional

import torch

from rllm.datasets import (
    TACM12KDataset,
    TLF2KDataset,
    TML1MDataset,
)
from rllm.data import HeteroGraphData, TableData


cache_dir = os.path.join(os.path.dirname(__file__), "data")
print("cache_dir:", cache_dir)


def _load_hgraph(dataset_name):
    tag_dir = os.path.join(os.path.dirname(__file__), "tag_data")
    file_path = os.path.join(tag_dir, dataset_name, dataset_name + '_graph.pkl')
    print("Load rllm.hgraph from: ", file_path)
    hgraph = HeteroGraphData.load(file_path)
    print(hgraph)
    if dataset_name == 'tacm12k':
        # hgraph['paper', 'cited_by', 'paper'].edge_index = hgraph['paper', 'cites', 'paper'].edge_index.flip(0)
        edge_index_ = torch.concat(
            [
                hgraph['paper', 'cites', 'paper'].edge_index,
                hgraph['paper', 'cites', 'paper'].edge_index.flip(0)
            ],
            dim=-1
        )
        hgraph['paper', 'cites', 'paper'].edge_index = edge_index_
        hgraph['author', 'writes', 'paper'].edge_index = hgraph['paper', 'written_by', 'author'].edge_index.flip(0)
    return hgraph


def load_dataset(dataset_name, load_graph: bool = False):
    r"""Load dataset based on the given name.
    Return a dictionary of tables.
    """
    if dataset_name == "tacm12k":
        dataset = TACM12KDataset(cache_dir)
        (
            paper_table,
            auther_table,
            citations_table,
            writings_table,
            paper_embeddings,
            author_embeddings,
        ) = dataset.data_list
        table_dict = {
            "papers": paper_table,
            "authors": auther_table,
            "citations": citations_table,
            "writings": writings_table,
            "paper_embeddings": paper_embeddings,
            "author_embeddings": author_embeddings,
        }
        if load_graph:
            g = _load_hgraph(dataset_name)
            return table_dict, g
        return table_dict

    elif dataset_name == "tlf2k":
        dataset = TLF2KDataset(cache_dir)
        (
            artists_table,
            user_artists_table,
            user_friends_table,
        ) = dataset.data_list
        return {
            "artists": artists_table,
            "user_artists": user_artists_table,
            "user_friends": user_friends_table,
        }

    elif dataset_name == "tml1m":
        dataset = TML1MDataset(cache_dir)
        (
            users_table,
            movies_table,
            ratings_table,
            movie_embeddings,
        ) = dataset.data_list
        return {
            "users": users_table,
            "movies": movies_table,
            "ratings": ratings_table,
            "movie_embeddings": movie_embeddings,
        }

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# t, g = load_dataset("tacm12k", load_graph=True)
# print(g)
# print(g.metadata())
