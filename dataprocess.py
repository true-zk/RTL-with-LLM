import os
import os.path as osp
import shutil
from typing import Optional, Union, Dict, Tuple, List, Generator
import json

import torch
from torch import Tensor
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from rllm.data import HeteroGraphData

from config import RAW_DATA_ROOT_DIR, TAG_ROOT_DIR, PROMPT_1_DIR, PROMPT_2_DIR
from llm import llm
from bge_model import embed_model, tokenizer
from utils import print_warning, print_success, print_danger


class BaseLoadCls:

    # data structure properties
    raw_data_list: list = []
    rag_data_list: list = []
    labels: list = []
    cnt_labels: int = 0

    # path properties
    dataset_name: str = ""

    # model properties
    llm = llm
    embed_model = embed_model
    tokenizer = tokenizer

    def __init__(self, rag_root_dir: str = TAG_ROOT_DIR) -> None:
        self.rag_dir = osp.join(rag_root_dir, self.dataset_name)
        self.prompt_set1_dir = osp.join(PROMPT_1_DIR, self.dataset_name)
        self.prompt_set2_dir = osp.join(PROMPT_2_DIR, self.dataset_name)
        if not osp.exists(self.rag_dir):
            os.makedirs(self.rag_dir, exist_ok=True)
        if not osp.exists(self.prompt_set1_dir):
            os.makedirs(self.prompt_set1_dir, exist_ok=True)
        if not osp.exists(self.prompt_set2_dir):
            os.makedirs(self.prompt_set2_dir, exist_ok=True)

        self.raw_dir = osp.join(RAW_DATA_ROOT_DIR, self.dataset_name)
        self.rag_dir = osp.join(TAG_ROOT_DIR, self.dataset_name)
        self.prompt_set1_dir = osp.join(PROMPT_1_DIR, self.dataset_name)

    def load_raw_df(self, file: Optional[str]) -> Union[pd.DataFrame, dict]:
        r"""Return the file; if file is None, return all files in raw data list."""
        if file is not None:
            assert file in self.raw_data_list
            file = osp.join(self.raw_dir, file)
            return pd.read_csv(file)

        data = {}
        for file in self.raw_data_list:
            file_path = osp.join(self.raw_dir, file)
            if not osp.exists(file_path):
                raise FileNotFoundError(f"File {file} not found in {self.raw_dir}")
            data[file] = pd.read_csv(file_path)
        return data

    def load_rag_data(self) -> Dict[str, Union[HeteroGraphData, str, torch.Tensor]]:
        r"""Load all data from rag data list."""
        data = {}
        for file in self.rag_data_list:
            file: str
            file_path = osp.join(self.rag_dir, file)
            if not osp.exists(file_path):
                raise FileNotFoundError(f"File {file} not found in {self.rag_dir}")
            if file.endswith(".pkl"):
                data['hgraph'] = HeteroGraphData.load(file_path)
            elif file.endswith(".txt"):
                with open(file_path, "r") as f:
                    data[file.replace(".txt", "")] = f.readlines()
            else:
                data[file.replace(".pt", "")] = torch.load(file_path, weights_only=False)
        return data

    def get_dataset(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        r"""Get dataset from rag data list."""
        data = self.load_rag_data()
        masks = data['masks']
        y = data['y']
        train_mask = masks['train_mask']
        val_mask = masks['val_mask']
        test_mask = masks['test_mask']
        train_ids = torch.nonzero(train_mask).view(-1)
        val_ids = torch.nonzero(val_mask).view(-1)
        test_ids = torch.nonzero(test_mask).view(-1)
        return [train_ids, val_ids, test_ids], y

    def get_pyg_graph(self) -> HeteroData:
        r"""Get pyg graph from rag data list."""
        data = self.load_rag_data()
        pyg_graph = data['pyg_graph']
        return pyg_graph

    def text_attr_fetch(self, texts: List[str], idx: Tensor) -> Generator:
        r"""Fetch text attrbutes from list."""
        idx = idx.tolist()
        for i in idx:
            yield texts[i]

    def gen_edge_prompt(self, edge_type: List[str], batch: HeteroData) -> str:
        edges: Tensor = batch[edge_type[0], edge_type[1], edge_type[2]].edge_index
        if edges.numel() == 0:
            return ""
        edge_prompt = ""
        for i in range(edges.shape[1]):
            edge_prompt += (
                f"『{edge_type[0]} node {edges[0][i]}』 "
                f"-- {edge_type[1]} --> "
                f"『{edge_type[2]} node {edges[1][i]}』\n"
            )
        return edge_prompt

    # abstract methods ########################################
    def build_tag(self) -> None:
        raise NotImplementedError("Please implement the build_tag method.")

    def pyg_loader(self, batch_size: int = 1) -> NeighborLoader:
        r"""Return a NeighborLoader for the pyg graph."""
        raise NotImplementedError("Please implement the pyg_loader method.")

    def prompt_set1_loader(self, batch_size: int = 1, persist: bool = True) -> Generator:
        r"""Return a generator for TAPE prompt."""
        raise NotImplementedError("Please implement the prompt_set1_loader method.")

    def prompt_set2_loader(self, batch_size: int = 1, persist: bool = True) -> Generator:
        r"""Return a generator for sampled TAPE prompt."""
        raise NotImplementedError("Please implement the prompt_set2_loader method.")


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

    def build_tag(self):
        # load author table
        author_df = self.load_raw_df("authors.csv")
        author_text_attr = author_df.apply(
            lambda row: (
                f"author_id is: {row['author_id']}, "
                f"name is: {row['name']}, "
                f"firm is: {row['firm']}."
            ),
            axis=1,
        )
        author_text_file = osp.join(self.rag_dir, "author_text.txt")
        with open(author_text_file, "w") as f:
            for text in author_text_attr:
                f.write(text + "\n")

        # load paper table
        # paper table is target table
        paper_df = self.load_raw_df("papers.csv")
        paper_text_attr = paper_df.apply(
            lambda row: (
                f"paper_id is: {row['paper_id']}, "
                f"year is: {row['year']}, "
                f"title is: {row['title']}, "
                f"abstract is: {row['abstract']}, "
                # f"conference is: {row['conference']}."  # y
            ),
            axis=1,
        )
        paper_text_file = osp.join(self.rag_dir, "paper_text.txt")
        with open(paper_text_file, "w") as f:
            for text in paper_text_attr:
                f.write(text + "\n")

        # y
        y = paper_df["conference"].values
        y = [self.labels.index(i) for i in y]
        y = torch.tensor(y, dtype=torch.long)
        torch.save(
            y,
            osp.join(self.rag_dir, "y.pt"),
        )

        # masks
        shutil.copy2(
            osp.join(self.raw_dir, "masks.pt"),
            osp.join(self.rag_dir, "masks.pt"),
        )

        hgraph = HeteroGraphData()
        pyg_graph = HeteroData()
        # Edges
        # load citation table
        citation_df = self.load_raw_df("citations.csv")
        edge_list = torch.tensor(
            citation_df[["paper_id", "paper_id_cited"]].values,
            dtype=torch.long,
        ).T
        hgraph['paper', 'cites', 'paper'].edge_index = edge_list
        # pyg_graph['paper', 'cites', 'paper'].edge_index = edge_list

        rev_edge_list = edge_list.flip(0)
        pyg_graph['paper', 'cited_by', 'paper'].edge_index = rev_edge_list  # for pyg loader

        # load writing table
        writing_df = self.load_raw_df("writings.csv")
        edge_list = torch.tensor(
            writing_df[["paper_id", "author_id"]].values,
            dtype=torch.long,
        ).T

        hgraph['paper', 'written_by', 'author'].edge_index = edge_list
        # pyg_graph['paper', 'written_by', 'author'].edge_index = edge_list

        rev_edge_list = edge_list.flip(0)
        pyg_graph['author', 'writes', 'paper'].edge_index = rev_edge_list  # for pyg loader

        # Nodes
        pyg_graph['paper'].num_nodes = hgraph['paper'].num_nodes = len(paper_df)
        pyg_graph['author'].num_nodes = hgraph['author'].num_nodes = len(author_df)
        # pyg_graph['paper'].text = paper_text_attr.tolist()
        pyg_graph['paper'].y = y
        # pyg_graph['author'].text = author_text_attr.tolist()

        print("rllm hgraph:", hgraph)
        print("pyg hgraph:", pyg_graph)
        hgraph.save(osp.join(self.rag_dir, "tacm12k_graph.pkl"))
        torch.save(pyg_graph, osp.join(self.rag_dir, "pyg_graph.pt"))

    # tape utility functions ##########################
    def pyg_loader(self, batch_size: int = 1) -> NeighborLoader:
        # Create a NeighborLoader for the PyG graph
        pyg_graph = self.get_pyg_graph()
        # target_node_id = torch.arange(len(y), dtype=torch.long)
        """
        pyg sampling for dst nodes:
        paper (neighbor) -> cited_by -> paper (seed)
        author (neighbor) -> writes -> paper (seed)
        """
        num_neighbors = {
            ('paper', 'cited_by', 'paper'): [5],
            ('author', 'writes', 'paper'): [5],
        }
        loader = NeighborLoader(
            pyg_graph,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=False,
            input_nodes="paper",
        )
        return loader

    def prompt_set1_loader(self, batch_size: int = 1, persist: bool = True) -> Generator:
        if persist:
            persist_dir = osp.join(self.prompt_set1_dir, f"{batch_size}")
            if not osp.exists(persist_dir):
                os.makedirs(persist_dir, exist_ok=True)

        prompt_temp = (
            "Here is a paper description: \n"
            "{paper_text} \n"
            "Question: Which conference is this paper published in? "
            "Give 5 likely conferences from {labels}. \n"
            "Answer format is like: [conference1, conference2, conference3, conference4, conference5]. \n"
            "And give your reason for the answer. \n"
            "Answer: \n"
            "Reason: \n"
        )
        prompt_temp_multi = (
            f"Here are {batch_size} paper descriptions: \n"
            "{paper_text} \n"
            "Question: Which conference are these paper published in separately? "
            "Give each paper 5 likely conferences from {labels}. \n"
            "Format of answer for each paper is like: [conference1, conference2, conference3, conference4, conference5]. \n"
            "And give your reason for each paper. \n"
            "Answers: \n"
            "Reasons: \n"
        )
        paper_text = self.load_rag_data()['paper_text']
        for i in range(0, len(paper_text), batch_size):
            if batch_size == 1:
                prompt = prompt_temp.format(
                    paper_text=paper_text[i],
                    labels=self.labels,
                )
            else:
                prompt = prompt_temp_multi.format(
                    paper_text="\n".join(paper_text[i:i + batch_size]),
                    labels=self.labels,
                )

            if persist:
                # Save the prompt to a file
                prompt_file = osp.join(persist_dir, f"prompt_{i // batch_size}.txt")
                with open(prompt_file, "w") as f:
                    f.write(prompt)
            yield prompt

    def prompt_set2_loader(self, batch_size: int = 1, persist: bool = True) -> Generator:
        r"""Return a generator for sampled TAPE prompt.
        If `persist`, save the prompts to 'prompt_set2_dir/{dataset_name}/{batch_size}.json'.
        """
        if persist:
            persist_dir = self.prompt_set2_dir
            if not osp.exists(persist_dir):
                os.makedirs(persist_dir, exist_ok=True)

        prompt_head = (
            "You are dealing with a small knowledge graph of nodes and edges."
            "Each node in the graph contains a text description, and "
            "the edge represents the relationship between nodes and the direction of information transmission."
            "The graph is: \n"
        )

        pyg_neighbor_loader = self.pyg_loader(batch_size)
        paper_text = self.load_rag_data()['paper_text']
        author_text = self.load_rag_data()['author_text']

        res = []
        for batch in pyg_neighbor_loader:
            num_nodes = f"Number of paper nodes: {batch['paper'].num_nodes}.\n"
            num_nodes += f"Number of author nodes: {batch['author'].num_nodes}.\n"

            nodes = "\n Paper nodes:\n"
            for j, el in enumerate(self.text_attr_fetch(paper_text, batch['paper'].n_id)):
                nodes += f"『Paper node {j}』, description: {el}\n"

            nodes += "\n Author nodes:\n"
            for j, el in enumerate(self.text_attr_fetch(author_text, batch['author'].n_id)):
                nodes += f"『Author node {j}』, description: {el}\n"

            edges = "\n Edges:\n"
            edges += self.gen_edge_prompt(
                ['paper', 'cited_by', 'paper'], batch
            )
            edges += self.gen_edge_prompt(
                ['author', 'writes', 'paper'], batch
            )

            prompt = prompt_head + num_nodes + nodes + edges
            yield prompt
            res.append(prompt)

        if persist:
            prompt_file = osp.join(persist_dir, f"{batch_size}.json")
            with open(prompt_file, "w") as f:
                json.dump(res, f)


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

    def build_tag(self):
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

        def parse_unknown(x, head):
            if pd.notna(x):
                return head + str(x) + ", "
            else:
                return head[:-2] + "not known, "

        artist_df = self.load_raw_df("artists.csv")
        artist_df['artistID'] = artist_df['artistID'] -1  # reindex
        artist_text_attr = artist_df.apply(
            lambda row: (
                f"artist_id is: {row['artistID']}, "
                f"{parse_type(row['type'])}"
                f"name is: {row['name']}, "
                f"{parse_born(row['born'])}"
                f"{parse_years_active(row['yearsActive'])}"
                f"{parse_unknown(row['location'], 'location is: ')}"
                f"{parse_unknown(row['biography'], 'biography is: ')}"
                f"personal url is: {row['url']}, "
                # f"label is: {row['label']}."
            ),
            axis=1,
        )
        artist_text_file = osp.join(self.rag_dir, "artist_text.txt")
        with open(artist_text_file, "w") as f:
            for text in artist_text_attr:
                f.write(text + "\n")

        # y
        y = artist_df["label"].values
        y = [self.labels.index(i.lower()) for i in y]
        y = torch.tensor(y, dtype=torch.long)
        torch.save(
            y,
            osp.join(self.rag_dir, "y.pt"),
        )
        # print(y.unique())

        shutil.copy2(
            osp.join(self.raw_dir, "masks.pt"),
            osp.join(self.rag_dir, "masks.pt"),
        )

        hgraph = HeteroGraphData()
        pyg_graph = HeteroData()
        # Edges
        # load user_friends table
        user_friends_df = self.load_raw_df("user_friends.csv")
        user_friends_df['userID'] = user_friends_df['userID'] - 1  # reindex
        user_friends_df['friendID'] = user_friends_df['friendID'] - 1
        # bi-directional edge
        edge_list = torch.tensor(
            user_friends_df[["userID", "friendID"]].values,
            dtype=torch.long,
        ).T
        rev_edge_list = edge_list.flip(0)
        edge_list = torch.cat([edge_list, rev_edge_list], dim=1)
        hgraph['user', 'friends_with', 'user'].edge_index = edge_list
        pyg_graph['user', 'friends_with', 'user'].edge_index = edge_list

        # load user_artists table
        user_artists_df = self.load_raw_df("user_artists.csv")
        user_artists_df['userID'] = user_artists_df['userID'] - 1  # reindex
        user_artists_df['artistID'] = user_artists_df['artistID'] - 1
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

        num_users = user_artists_df['userID'].max() + 1 # 1892, userID is 0-1891

        pyg_graph['user'].num_nodes = hgraph['user'].num_nodes = num_users  # TODO
        pyg_graph['artist'].num_nodes = hgraph['artist'].num_nodes = len(artist_df)

        # pyg_graph['artist'].text = artist_text_attr.tolist()
        pyg_graph['artist'].y = y
        # pyg_graph['user'].text = ["UserId is " + str(i) for i in range(num_users)]

        print(hgraph)
        print(pyg_graph)
        hgraph.save(osp.join(self.rag_dir, "tlf2k_graph.pkl"))
        torch.save(pyg_graph, osp.join(self.rag_dir, "pyg_graph.pt"))

    # sampled tape utility functions ##########################
    def pyg_loader(self, batch_size: int = 1) -> NeighborLoader:
        # Create a NeighborLoader for the PyG graph
        pyg_graph = self.get_pyg_graph()
        # target_node_id = torch.arange(len(y), dtype=torch.long)
        num_neighbors = {
            ('user', 'friends_with', 'user'): [5],
            ('user', 'likes', 'artist'): [5],
        }
        loader = NeighborLoader(
            pyg_graph,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=False,
            input_nodes="artist",
        )
        return loader

    def prompt_set1_loader(self, batch_size: int = 1, persist: bool = True) -> Generator:
        if persist:
            persist_dir = osp.join(self.prompt_set1_dir, f"{batch_size}")
            if not osp.exists(persist_dir):
                os.makedirs(persist_dir, exist_ok=True)

        prompt_temp = (
            "Here is a music artist's description: \n"
            "{artist_text} \n"
            "Question: Which genre does this artist belong to? "
            "Give 5 likely genres from {labels}. \n"
            "Answer format is like: [genre1, genre2, genre3, genre4, genre5]. \n"
            "And give your reason for the answer. \n"
            "Answer: \n"
            "Reason: \n"
        )
        prompt_temp_multi = (
            f"Here are {batch_size} music artists' descriptions: \n"
            "{artist_text} \n"
            "Question: Which genres do these artists belong to separately? "
            "Give 5 likely genres from {labels} separately. \n"
            "Format of answer for each artist is like: [genre1, genre2, genre3, genre4, genre5]. \n"
            "And give your reason for each artist. \n"
            "Answers: \n"
            "Reasons: \n"
        )
        artist_text = self.load_rag_data()['artist_text']
        for i in range(0, len(artist_text), batch_size):
            if batch_size == 1:
                prompt = prompt_temp.format(
                    artist_text=artist_text[i],
                    labels=self.labels,
                )
            else:
                prompt = prompt_temp_multi.format(
                    artist_text="\n".join(artist_text[i:i + batch_size]),
                    labels=self.labels,
                )

            if persist:
                # Save the prompt to a file
                prompt_file = osp.join(persist_dir, f"prompt_{i // batch_size}.txt")
                with open(prompt_file, "w") as f:
                    f.write(prompt)
            yield prompt

    def prompt_set2_loader(self, batch_size: int = 1, persist: bool = True) -> Generator:
        # TODO 只采样一阶邻居不够，需要采样friends然后再采样一阶artist
        r"""Return a generator for sampled TAPE prompt.
        If `persist`, save the prompts to 'prompt_set2_dir/{dataset_name}/{batch_size}.json'.
        """
        if persist:
            persist_dir = self.prompt_set2_dir
            if not osp.exists(persist_dir):
                os.makedirs(persist_dir, exist_ok=True)

        prompt_head = (
            "You are dealing with a small knowledge graph of nodes and edges."
            "Each node in the graph contains a text description, and "
            "the edge represents the relationship between nodes and the direction of information transmission."
            "The graph is: \n"
        )

        prompt_tail = (
            ""
        )

        pyg_neighbor_loader = self.pyg_loader(batch_size)
        artist_text = self.load_rag_data()['artist_text']

        res = []
        for batch in pyg_neighbor_loader:
            num_nodes = f"Number of artist nodes: {batch['artist'].num_nodes}.\n"
            num_nodes += f"Number of user nodes: {batch['user'].num_nodes}.\n"

            nodes = "\n Paper nodes:\n"
            for j, el in enumerate(self.text_attr_fetch(paper_text, batch['paper'].n_id)):
                nodes += f"『Paper node {j}』, description: {el}\n"

            nodes += "\n Author nodes:\n"
            for j, el in enumerate(self.text_attr_fetch(author_text, batch['author'].n_id)):
                nodes += f"『Author node {j}』, description: {el}\n"

            edges = "\n Edges:\n"
            edges += self.gen_edge_prompt(
                ['paper', 'cited_by', 'paper'], batch
            )
            edges += self.gen_edge_prompt(
                ['author', 'writes', 'paper'], batch
            )

            prompt = prompt_head + num_nodes + nodes + edges
            yield prompt
            res.append(prompt)

        if persist:
            prompt_file = osp.join(persist_dir, f"{batch_size}.json")
            with open(prompt_file, "w") as f:
                json.dump(res, f)

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

    def build_tag(self):
        # load user table
        user_df = self.load_raw_df("users.csv")
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
        user_text_file = osp.join(self.rag_dir, "user_text.txt")
        with open(user_text_file, "w") as f:
            for text in text_attr:
                f.write(text + "\n")

        def parse_unknown(x, head):
            if pd.notna(x):
                return head + str(x) + ", "
            else:
                return head[:-2] + "not known, "

        # load movie table
        movie_df = self.load_raw_df("movies.csv")
        text_attr = movie_df.apply(
            lambda row: (
                f"MovieID is: {row['MovieID']}, "
                f"Title is: {row['Title']}, "
                f"{parse_unknown(row['Year'], 'Year is: ')}"
                f"Genres are: {row['Genre']}, "
                f"{parse_unknown(row['Director'], 'Director is: ')}"
                f"{parse_unknown(row['Cast'], 'Cast is: ')}"
                f"{parse_unknown(row['Runtime'], 'Runtime is: ')}"
                f"{parse_unknown(row['Languages'], 'Languages is: ')}"
                f"{parse_unknown(row['Certificate'], 'Certificate code is: ')}"
                f"{parse_unknown(row['Plot'], 'Plot is: ')}"
                f"Url is: {row['Url']}."
            ),
            axis=1,
        )
        movie_text_file = osp.join(self.rag_dir, "movie_text.txt")
        with open(movie_text_file, "w") as f:
            for text in text_attr:
                f.write(text + "\n")

        # y
        y = user_df["Age"].values
        y = [self.labels.index(i) for i in y]
        y = torch.tensor(y, dtype=torch.long)
        torch.save(
            y,
            osp.join(self.rag_dir, "y.pt"),
        )
        # print(y.unique())

        # masks
        shutil.copy2(
            osp.join(self.raw_dir, "masks.pt"),
            osp.join(self.rag_dir, "masks.pt"),
        )

        hgraph = HeteroGraphData()
        pyg_graph = HeteroData()
        # Edges
        # load ratings table
        ratings_df = self.load_raw_df("ratings.csv")
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

        pyg_graph['user'].y = y

        print(hgraph)
        print(pyg_graph)
        hgraph.save(osp.join(self.rag_dir, "tml1m_graph.pkl"))
        torch.save(pyg_graph, osp.join(self.rag_dir, "pyg_graph.pt"))

    # sampled tape utility functions ##########################
    def pyg_loader(self, batch_size: int = 1) -> NeighborLoader:
        # Create a NeighborLoader for the PyG graph
        pyg_graph = self.get_pyg_graph()
        _, y = self.get_dataset()
        # target_node_id = torch.arange(len(y), dtype=torch.long)
        num_neighbors = {
            ('user', 'rates', 'movie'): [10, 5],
        }
        loader = NeighborLoader(
            pyg_graph,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=False,
            input_nodes="user",
        )
        return loader

    def prompt_set1_loader(self, batch_size: int = 1, persist: bool = True) -> Generator:
        print_warning(
            "user table in TML1M contains very little information, "
            "so this prompt may not be very useful. "
        )
        if persist:
            persist_dir = osp.join(self.prompt_set1_dir, f"{batch_size}")
            if not osp.exists(persist_dir):
                os.makedirs(persist_dir, exist_ok=True)

        prompt_temp = (
            "Here is a moive rating website user's description: \n"
            "{user_text} \n"
            "Question: What is the user's age? "
            "Give 5 likely age from {labels}. \n"
            "Answer format is like: [age1, age2, age3, age4, age5]. \n"
            "And give your reason for the answer. \n"
            "Answer: \n"
            "Reason: \n"
        )
        prompt_temp_multi = (
            f"Here are {batch_size} moive rating website userss descriptions: \n"
            "{user_text} \n"
            "Question: What are these users' age separately? "
            "Give 5 likely age from {labels} separately. \n"
            "Format of answer for each user is like: [age1, age2, age3, age4, age5]. \n"
            "And give your reason for each user. \n"
            "Answers: \n"
            "Reasons: \n"
        )
        user_text = self.load_rag_data()['user_text']
        for i in range(0, len(user_text), batch_size):
            if batch_size == 1:
                prompt = prompt_temp.format(
                    user_text=user_text[i],
                    labels=self.labels,
                )
            else:
                prompt = prompt_temp_multi.format(
                    user_text="\n".join(user_text[i:i + batch_size]),
                    labels=self.labels,
                )

            if persist:
                # Save the prompt to a file
                prompt_file = osp.join(persist_dir, f"prompt_{i // batch_size}.txt")
                with open(prompt_file, "w") as f:
                    f.write(prompt)
            yield prompt


# Script functions ############################################

datasets = [TACM12K, TLF2K, TML1M]

def apply_all(fn: str, *args, **kwargs):
    r"""Apply a function to all datasets."""
    for cls in datasets:
        ins = cls()
        if hasattr(ins, fn):
            method = getattr(ins, fn)
            if callable(method):
                print(f"Applying {fn} to {cls.__name__} ...")
                method(*args, **kwargs)
            else:
                print_danger(f"{fn} is not a callable method.")
                raise ValueError(f"{fn} is not a callable method.")
    print_success(
        f"Applied {fn} to all datasets."
    )

def build_tag():
    apply_all("build_tag")

def build_prompt1():
    r"""Build all datasets."""
    for i in [1, 5, 10]:
        apply_all("prompt_set1_loader", i, True)


a = TACM12K()
# a.build_tag()
for i in a.prompt_set2_loader():
    continue

with open("prompt_2/tacm12k/1.json", "r") as f:
    data = json.load(f)
print(len(data))
print(data[0])