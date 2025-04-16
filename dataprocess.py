import os
import os.path as osp
import shutil
from typing import Optional, Union, Dict, Tuple, List

import torch
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from rllm.data import HeteroGraphData

from config import RAW_DATA_ROOT_DIR, RAG_ROOT_DIR
from llm import llm
# from bge_model import embed_model, tokenizer


class BaseLoadCls:

    # data structure properties
    raw_data_list: list = []
    rag_data_list: list = []
    labels: list = []
    cnt_labels: int = 0

    # path properties
    dataset_name: str = ""
    raw_dir: str = ""
    rag_dir: str = ""

    # model properties
    llm = llm
    # embed_model = embed_model
    # tokenizer = tokenizer

    @classmethod
    def load_raw_df(cls, file: Optional[str]) -> Union[pd.DataFrame, dict]:
        r"""Return the file; if file is None, return all files in raw data list."""
        if file is not None:
            assert file in cls.raw_data_list
            file = osp.join(cls.raw_dir, file)
            return pd.read_csv(file)

        data = {}
        for file in cls.raw_data_list:
            file_path = osp.join(cls.raw_dir, file)
            if not osp.exists(file_path):
                raise FileNotFoundError(f"File {file} not found in {cls.raw_dir}")
            data[file] = pd.read_csv(file_path)
        return data

    @classmethod
    def load_rag_data(cls) -> Dict[str, Union[HeteroGraphData, str, torch.Tensor]]:
        r"""Load all data from rag data list."""
        data = {}
        for file in cls.rag_data_list:
            file: str
            file_path = osp.join(cls.rag_dir, file)
            if not osp.exists(file_path):
                raise FileNotFoundError(f"File {file} not found in {cls.rag_dir}")
            if file.endswith(".pkl"):
                data['hgraph'] = HeteroGraphData.load(file_path)
            elif file.endswith(".txt"):
                with open(file_path, "r") as f:
                    data[file.replace(".txt", "")] = f.readlines()
            else:
                data[file.replace(".pt", "")] = torch.load(file_path, weights_only=False)
        return data

    @classmethod
    def get_dataset(cls) -> Tuple[List[torch.Tensor], torch.Tensor]:
        r"""Get dataset from rag data list."""
        data = cls.load_rag_data()
        masks = data['masks']
        y = data['y']
        train_mask = masks['train_mask']
        val_mask = masks['val_mask']
        test_mask = masks['test_mask']
        train_ids = torch.nonzero(train_mask).view(-1)
        val_ids = torch.nonzero(val_mask).view(-1)
        test_ids = torch.nonzero(test_mask).view(-1)
        return [train_ids, val_ids, test_ids], y

    @classmethod
    def get_pyg_graph(cls) -> HeteroData:
        r"""Get pyg graph from rag data list."""
        data = cls.load_rag_data()
        pyg_graph = data['pyg_graph']
        return pyg_graph

    # abstract methods ########################################
    def build_tag(self) -> None:
        raise NotImplementedError("Please implement the build_tag method.")


class TACM12K(BaseLoadCls):

    raw_data_list = [
        "authors.csv",
        "papers.csv",
        "writings.csv",
        "citations.csv",
        "masks.pt",
    ]
    rag_data_list = [
        "tacm12k_graph.pkl",
        "author_text.txt",
        "paper_text.txt",
        "y.pt",
        "pyg_graph.pt",
        "masks.pt",
    ]
    labels = [
        'KDD', 'CIKM', 'WWW', 'SIGIR', 'STOC', 'MobiCOMM', 'SIGMOD', 'SIGCOMM', 'SPAA',
        'ICML', 'VLDB', 'SOSP', 'SODA', 'COLT'
    ]
    cnt_labels = len(labels)  # 14

    dataset_name = "tacm12k"
    raw_dir = osp.join(RAW_DATA_ROOT_DIR, dataset_name)
    rag_dir = osp.join(RAG_ROOT_DIR, dataset_name)

    def __init__(self, rag_root_dir: str = RAG_ROOT_DIR) -> None:
        self.rag_dir = osp.join(rag_root_dir, self.dataset_name)
        if not osp.exists(self.rag_dir):
            os.makedirs(self.rag_dir, exist_ok=True)

    @classmethod
    def build_tag(cls):
        # load author table
        author_df = cls.load_raw_df("authors.csv")
        text_attr = author_df.apply(
            lambda row: (
                f"author_id is: {row['author_id']}, "
                f"name is: {row['name']}, "
                f"firm is: {row['firm']}."
            ),
            axis=1,
        )
        author_text_file = osp.join(cls.rag_dir, "author_text.txt")
        with open(author_text_file, "w") as f:
            for text in text_attr:
                f.write(text + "\n")

        # load paper table
        # paper table is target table
        paper_df = cls.load_raw_df("papers.csv")
        text_attr = paper_df.apply(
            lambda row: (
                f"paper_id is: {row['paper_id']}, "
                f"year is: {row['year']}, "
                f"title is: {row['title']}, "
                f"abstract is: {row['abstract']}, "
                # f"conference is: {row['conference']}."  # y
            ),
            axis=1,
        )
        paper_text_file = osp.join(cls.rag_dir, "paper_text.txt")
        with open(paper_text_file, "w") as f:
            for text in text_attr:
                f.write(text + "\n")

        # y
        y = paper_df["conference"].values
        y = [cls.labels.index(i) for i in y]
        y = torch.tensor(y, dtype=torch.long)
        torch.save(
            y,
            osp.join(cls.rag_dir, "y.pt"),
        )

        # masks
        shutil.copy2(
            osp.join(cls.raw_dir, "masks.pt"),
            osp.join(cls.rag_dir, "masks.pt"),
        )

        hgraph = HeteroGraphData()
        pyg_graph = HeteroData()
        # Edges
        # load citation table
        citation_df = cls.load_raw_df("citations.csv")
        edge_list = torch.tensor(
            citation_df[["paper_id", "paper_id_cited"]].values,
            dtype=torch.long,
        ).T
        hgraph['paper', 'cites', 'paper'].edge_index = edge_list
        pyg_graph['paper', 'cites', 'paper'].edge_index = edge_list

        # load writing table
        writing_df = cls.load_raw_df("writings.csv")
        edge_list = torch.tensor(
            writing_df[["paper_id", "author_id"]].values,
            dtype=torch.long,
        ).T
        hgraph['paper', 'written_by', 'author'].edge_index = edge_list
        pyg_graph['paper', 'written_by', 'author'].edge_index = edge_list

        # Nodes
        pyg_graph['paper'].num_nodes = hgraph['paper'].num_nodes = len(paper_df)
        pyg_graph['author'].num_nodes = hgraph['author'].num_nodes = len(author_df)

        print("rllm hgraph:", hgraph)
        print("pyg hgraph:", pyg_graph)
        hgraph.save(osp.join(cls.rag_dir, "tacm12k_graph.pkl"))
        torch.save(pyg_graph, osp.join(cls.rag_dir, "pyg_graph.pt"))

    # sampled tape utility functions ##########################
    @classmethod
    def pyg_loader(cls, batch_size: int = 1) -> NeighborLoader:
        # Create a NeighborLoader for the PyG graph
        pyg_graph = cls.get_pyg_graph()
        _, y = cls.get_dataset()
        target_node_id = torch.arange(len(y), dtype=torch.long)
        num_neighbors = {
            ('paper', 'cites', 'paper'): [10, 5],
            ('paper', 'written_by', 'author'): [10, 5],
        }
        loader = NeighborLoader(
            pyg_graph,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=False,
            input_nodes=target_node_id,
        )
        return loader


class TLF2K(BaseLoadCls):
    raw_data_list = [
        "artists.csv",
        "user_artists.csv",
        "user_friends.csv",
        "masks.pt",
    ]
    rag_data_list = [
        "tlf2k_graph.pkl",
        "artist_text.txt",
        "y.pt",
        "pyg_graph.pt",
        "masks.pt"
    ]
    labels = [
        'Country', 'Electronic', 'Hip-Hop', 'Jazz', 'Latin', 'Pop',
        'Punk', 'Reggae', 'Rock', 'Metal', 'Soul'
    ]
    labels = [i.lower() for i in labels]
    cnt_labels = len(labels)  # 11

    dataset_name = "tlf2k"
    raw_dir = osp.join(RAW_DATA_ROOT_DIR, dataset_name)
    rag_dir = osp.join(RAG_ROOT_DIR, dataset_name)

    def __init__(self, rag_root_dir: str = RAG_ROOT_DIR) -> None:
        self.rag_dir = osp.join(rag_root_dir, self.dataset_name)
        if not osp.exists(self.rag_dir):
            os.makedirs(self.rag_dir, exist_ok=True)

    def build_tag(cls):
        # load artist table
        def parse_type(x):
            if x == 'single':
                return "this artist is individual, "
            elif x == 'group':
                return "this artist is a group, "
            else:
                return "nan, "

        def parse_born(x):
            if isinstance(x, str):
                return f"born in {x}, "
            else:
                return "birth is unknown, "

        def parse_years_active(x):
            if isinstance(x, str):
                return f"the group's years active is {x}, "
            else:
                return "years active is unknown, "

        artist_df = cls.load_raw_df("artists.csv")
        text_attr = artist_df.apply(
            lambda row: (
                f"artist_id is: {row['artistID']}, "
                + parse_type(row['type']) +
                f"name is: {row['name']}, "
                + parse_born(row['born'])
                + parse_years_active(row['yearsActive']) +
                f"location is: {row['location']}, " if pd.notna(row['location']) else "localtion is not known, "
                f"biography is: {row['biography']}, " if pd.notna(row['biography']) else "biography is not known, "
                f"personal url is: {row['url']}, "
                # f"label is: {row['label']}."
            ),
            axis=1,
        )
        artist_text_file = osp.join(cls.rag_dir, "artist_text.txt")
        with open(artist_text_file, "w") as f:
            for text in text_attr:
                f.write(text + "\n")

        # y
        y = artist_df["label"].values
        y = [cls.labels.index(i.lower()) for i in y]
        y = torch.tensor(y, dtype=torch.long)
        torch.save(
            y,
            osp.join(cls.rag_dir, "y.pt"),
        )
        # print(y.unique())

        shutil.copy2(
            osp.join(cls.raw_dir, "masks.pt"),
            osp.join(cls.rag_dir, "masks.pt"),
        )

        hgraph = HeteroGraphData()
        pyg_graph = HeteroData()
        # Edges
        # load user_friends table
        user_friends_df = cls.load_raw_df("user_friends.csv")
        # bi-directional edge
        edge_list = torch.tensor(
            user_friends_df[["userID", "friendID"]].values,
            dtype=torch.long,
        ).T
        # Keep this process when using
        # rev_edge_list = edge_list.flip(0)
        # edge_list = torch.cat([edge_list, rev_edge_list], dim=1)
        hgraph['user', 'friends_with', 'user'].edge_index = edge_list
        pyg_graph['user', 'friends_with', 'user'].edge_index = edge_list

        # load user_artists table
        user_artists_df = cls.load_raw_df("user_artists.csv")
        edge_list = torch.tensor(
            user_artists_df[["userID", "artistID"]].values,
            dtype=torch.long,
        ).T
        hgraph['user', 'likes', 'artist'].edge_index = edge_list
        pyg_graph['user', 'likes', 'artist'].edge_index = edge_list
        edge_weight = torch.tensor(
            user_artists_df["weight"].values,
            dtype=torch.long,
        )
        hgraph['user', 'likes', 'artist'].listening_cnt = edge_weight
        pyg_graph['user', 'likes', 'artist'].listening_cnt = edge_weight

        pyg_graph['user'].num_nodes = hgraph['user'].num_nodes = len(user_artists_df)  # TODO
        pyg_graph['artist'].num_nodes = hgraph['artist'].num_nodes = len(artist_df)

        print(hgraph)
        hgraph.save(osp.join(cls.rag_dir, "tlf2k_graph.pkl"))
        torch.save(pyg_graph, osp.join(cls.rag_dir, "pyg_graph.pt"))

    # sampled tape utility functions ##########################
    @classmethod
    def pyg_loader(cls, batch_size: int = 1) -> NeighborLoader:
        # Create a NeighborLoader for the PyG graph
        pyg_graph = cls.get_pyg_graph()
        _, y = cls.get_dataset()
        target_node_id = torch.arange(len(y), dtype=torch.long)
        num_neighbors = {
            ('user', 'friends_with', 'user'): [10, 5],
            ('user', 'likes', 'artist'): [10, 5],
        }
        loader = NeighborLoader(
            pyg_graph,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=False,
            input_nodes=target_node_id,
        )
        return loader


class TML1M(BaseLoadCls):
    raw_data_list = [
        "users.csv",
        "movies.csv",
        "ratings.csv",
        "masks.pt",
    ]
    rag_data_list = [
        "tml1m_graph.pkl",
        "user_text.txt",
        "movie_text.txt",
        "y.pt",
        "pyg_graph.pt",
        "masks.pt"
    ]
    labels = [1, 18, 25, 35, 45, 50, 56]
    cnt_labels = len(labels)  # 7

    dataset_name = "tml1m"
    raw_dir = osp.join(RAW_DATA_ROOT_DIR, dataset_name)
    rag_dir = osp.join(RAG_ROOT_DIR, dataset_name)

    def __init__(self, rag_root_dir: str = RAG_ROOT_DIR) -> None:
        self.rag_dir = osp.join(rag_root_dir, self.dataset_name)
        if not osp.exists(self.rag_dir):
            os.makedirs(self.rag_dir, exist_ok=True)

    @classmethod
    def build_tag(cls):
        # load user table
        user_df = cls.load_raw_df("users.csv")
        text_attr = user_df.apply(
            lambda row: (
                f"UserID is: {row['UserID']}, "
                f"Gender is: {row['Gender']},  "
                f"Occupation id is: {row['Occupation']}, "
                f"Zip-code is: {row['Zip-code']}, "
                # f"Age is: {row['Age']}."  # y
            ),
            axis=1,
        )
        user_text_file = osp.join(cls.rag_dir, "user_text.txt")
        with open(user_text_file, "w") as f:
            for text in text_attr:
                f.write(text + "\n")

        # load movie table
        movie_df = cls.load_raw_df("movies.csv")
        text_attr = movie_df.apply(
            lambda row: (
                f"MovieID is: {row['MovieID']}, "
                f"Title is: {row['Title']}, "
                f"Year is: {row['Year']}, " if pd.notna(row['Year']) else "Year is not known, "
                f"Genres are: {row['Genres']}, "
                f"Director is: {row['Director']}, " if pd.notna(row['Director']) else "Director is not known, "
                f"Cast is: {row['Cast']}, " if pd.notna(row['Cast']) else "Cast is not known, "
                f"Runtime is: {row['Runtime']}, " if pd.notna(row['Runtime']) else "Runtime is not known, "
                f"Languages is: {row['Languages']}, " if pd.notna(row['Languages']) else "Languages is not known, "
                f"Certificate code is: {row['Certificate']}, " if pd.notna(row['Certificate'])
                else "Certificate code is not known, "
                f"Plot is: {row['Plot']}, " if pd.notna(row['Plot']) else "Plot is not known, "
                f"Url is: {row['Url']}."
            ),
            axis=1,
        )
        movie_text_file = osp.join(cls.rag_dir, "movie_text.txt")
        with open(movie_text_file, "w") as f:
            for text in text_attr:
                f.write(text + "\n")

        # y
        y = user_df["Age"].values
        y = [cls.labels.index(i) for i in y]
        y = torch.tensor(y, dtype=torch.long)
        torch.save(
            y,
            osp.join(cls.rag_dir, "y.pt"),
        )
        # print(y.unique())

        # masks
        shutil.copy2(
            osp.join(cls.raw_dir, "masks.pt"),
            osp.join(cls.rag_dir, "masks.pt"),
        )

        hgraph = HeteroGraphData()
        pyg_graph = HeteroData()
        # Edges
        # load ratings table
        ratings_df = cls.load_raw_df("ratings.csv")
        edge_list = torch.tensor(
            ratings_df[["UserID", "MovieID"]].values,
            dtype=torch.long,
        ).T
        hgraph['user', 'rates', 'movie'].edge_index = edge_list
        pyg_graph['user', 'rates', 'movie'].edge_index = edge_list
        edge_weight = torch.tensor(
            ratings_df["Rating"].values,
            dtype=torch.float,
        )
        hgraph['user', 'rates', 'movie'].rating = edge_weight
        pyg_graph['user', 'rates', 'movie'].rating = edge_weight
        time_stamp = torch.tensor(
            ratings_df["Timestamp"].values,
            dtype=torch.long,
        )
        hgraph['user', 'rates', 'movie'].time_stamp = time_stamp
        pyg_graph['user', 'rates', 'movie'].time_stamp = time_stamp

        pyg_graph['user'].num_nodes = hgraph['user'].num_nodes = len(user_df)
        pyg_graph['movie'].num_nodes = hgraph['movie'].num_nodes = len(movie_df)

        print(hgraph)
        hgraph.save(osp.join(cls.rag_dir, "tml1m_graph.pkl"))
        torch.save(pyg_graph, osp.join(cls.rag_dir, "pyg_graph.pt"))

    # sampled tape utility functions ##########################
    @classmethod
    def pyg_loader(cls, batch_size: int = 1) -> NeighborLoader:
        # Create a NeighborLoader for the PyG graph
        pyg_graph = cls.get_pyg_graph()
        _, y = cls.get_dataset()
        target_node_id = torch.arange(len(y), dtype=torch.long)
        num_neighbors = {
            ('user', 'rates', 'movie'): [10, 5],
        }
        loader = NeighborLoader(
            pyg_graph,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=False,
            input_nodes=target_node_id,
        )
        return loader


TACM12K.build_tag()
loader = TACM12K.pyg_loader(batch_size=2)
for batch in loader:
    # batch is a dict with keys 'paper', 'author', and 'user'
    # each key contains a batch of nodes and their neighbors
    print(batch)
    break
# g = tacm.pyg_graph()

# tacm.build_tag()

# tlf2k = TLF2K()
# tlf2k.build_tag()

# tml1m = TML1M()
# tml1m.build_tag()

# TACM12K.load_all()
