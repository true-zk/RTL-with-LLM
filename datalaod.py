import os
import os.path as osp
import shutil
from typing import Optional, Union

import torch
import pandas as pd

from rllm.data import HeteroGraphData

from config import RAW_DATA_ROOT_DIR, RAG_ROOT_DIR


class BaseLoadCls:

    raw_data_list: list = []
    dataset_name: str = ""
    raw_dir: str = ""
    rag_dir: str = ""

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

    def build_tag(self):
        raise NotImplementedError("Please implement the build_tag method.")


class TACM12K(BaseLoadCls):

    raw_data_list = [
        "authors.csv",
        "papers.csv",
        "writings.csv",
        "citations.csv",
        "masks.pt",
    ]

    dataset_name = "tacm12k"
    raw_dir = osp.join(RAW_DATA_ROOT_DIR, dataset_name)

    def __init__(self, rag_root_dir: str = RAG_ROOT_DIR) -> None:
        self.rag_dir = osp.join(rag_root_dir, self.dataset_name)
        if not osp.exists(self.rag_dir):
            os.makedirs(self.rag_dir, exist_ok=True)

    def build_tag(self):
        hgraph = HeteroGraphData()
        # Edges
        # load citation table
        citation_df = self.load_raw_df("citations.csv")
        edge_list = torch.tensor(
            citation_df[["paper_id", "paper_id_cited"]].values,
            dtype=torch.long,
        ).T
        hgraph['paper', 'cites', 'paper'].edge_index = edge_list

        # load writing table
        writing_df = self.load_raw_df("writings.csv")
        edge_list = torch.tensor(
            writing_df[["paper_id", "author_id"]].values,
            dtype=torch.long,
        ).T
        hgraph['paper', 'written_by', 'author'].edge_index = edge_list
        print(hgraph)
        hgraph.save(osp.join(self.rag_dir, "tacm12k_graph.pkl"))

        # Node attributes
        # load author table
        author_df = self.load_raw_df("authors.csv")
        text_attr = author_df.apply(
            lambda row: (
                f"author_id is: {row['author_id']}, "
                f"name is: {row['name']}, "
                f"firm is: {row['firm']}."
            ),
            axis=1,
        )
        author_text_file = osp.join(self.rag_dir, "author_text.txt")
        with open(author_text_file, "w") as f:
            for text in text_attr:
                f.write(text + "\n")

        # load paper table
        # paper table is target table
        paper_df = self.load_raw_df("papers.csv")
        text_attr = paper_df.apply(
            lambda row: (
                f"paper_id is: {row['paper_id']}, "
                f"year is: {row['year']}, "
                f"title is: {row['title']}, "
                f"abstract is: {row['abstract']}, "
                f"conference is: {row['conference']}."
            ),
            axis=1,
        )
        paper_text_file = osp.join(self.rag_dir, "paper_text.txt")
        with open(paper_text_file, "w") as f:
            for text in text_attr:
                f.write(text + "\n")

        shutil.copy2(
            osp.join(self.raw_dir, "masks.pt"),
            osp.join(self.rag_dir, "masks.pt"),
        )


class TLF2K(BaseLoadCls):
    raw_data_list = [
        "artists.csv",
        "user_artists.csv",
        "user_friends.csv",
        "masks.pt",
    ]

    dataset_name = "tlf2k"
    raw_dir = osp.join(RAW_DATA_ROOT_DIR, dataset_name)

    def __init__(self, rag_root_dir: str = RAG_ROOT_DIR) -> None:
        self.rag_dir = osp.join(rag_root_dir, self.dataset_name)
        if not osp.exists(self.rag_dir):
            os.makedirs(self.rag_dir, exist_ok=True)

    def build_tag(self):
        hgraph = HeteroGraphData()
        # Edges
        # load user_friends table
        user_friends_df = self.load_raw_df("user_friends.csv")
        # bi-directional edge
        edge_list = torch.tensor(
            user_friends_df[["userID", "friendID"]].values,
            dtype=torch.long,
        ).T
        # Keep this process when using
        # rev_edge_list = edge_list.flip(0)
        # edge_list = torch.cat([edge_list, rev_edge_list], dim=1)
        hgraph['user', 'friends_with', 'user'].edge_index = edge_list

        # load user_artists table
        user_artists_df = self.load_raw_df("user_artists.csv")
        edge_list = torch.tensor(
            user_artists_df[["userID", "artistID"]].values,
            dtype=torch.long,
        ).T
        hgraph['user', 'likes', 'artist'].edge_index = edge_list
        edge_weight = torch.tensor(
            user_artists_df["weight"].values,
            dtype=torch.long,
        )
        hgraph['user', 'likes', 'artist'].listening_cnt = edge_weight

        print(hgraph)
        hgraph.save(osp.join(self.rag_dir, "tlf2k_graph.pkl"))

        # Node attributes
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

        artist_df = self.load_raw_df("artists.csv")
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
                f"label is: {row['label']}."
            ),
            axis=1,
        )
        artist_text_file = osp.join(self.rag_dir, "artist_text.txt")
        with open(artist_text_file, "w") as f:
            for text in text_attr:
                f.write(text + "\n")

        shutil.copy2(
            osp.join(self.raw_dir, "masks.pt"),
            osp.join(self.rag_dir, "masks.pt"),
        )


class TML1M(BaseLoadCls):
    raw_data_list = [
        "users.csv",
        "movies.csv",
        "ratings.csv",
        "masks.pt",
    ]

    dataset_name = "tml1m"
    raw_dir = osp.join(RAW_DATA_ROOT_DIR, dataset_name)

    def __init__(self, rag_root_dir: str = RAG_ROOT_DIR) -> None:
        self.rag_dir = osp.join(rag_root_dir, self.dataset_name)
        if not osp.exists(self.rag_dir):
            os.makedirs(self.rag_dir, exist_ok=True)

    def build_tag(self):
        hgraph = HeteroGraphData()
        # Edges
        # load ratings table
        ratings_df = self.load_raw_df("ratings.csv")
        edge_list = torch.tensor(
            ratings_df[["UserID", "MovieID"]].values,
            dtype=torch.long,
        ).T
        hgraph['user', 'rates', 'movie'].edge_index = edge_list
        edge_weight = torch.tensor(
            ratings_df["Rating"].values,
            dtype=torch.float,
        )
        hgraph['user', 'rates', 'movie'].rating = edge_weight
        time_stamp = torch.tensor(
            ratings_df["Timestamp"].values,
            dtype=torch.long,
        )
        hgraph['user', 'rates', 'movie'].time_stamp = time_stamp

        print(hgraph)
        hgraph.save(osp.join(self.rag_dir, "tml1m_graph.pkl"))

        # Node attributes
        # load user table
        user_df = self.load_raw_df("users.csv")
        text_attr = user_df.apply(
            lambda row: (
                f"UserID is: {row['UserID']}, "
                f"Gender is: {row['Gender']},  "
                f"Occupation id is: {row['Occupation']}, "
                f"Zip-code is: {row['Zip-code']}, "
                f"Age is: {row['Age']}."
            ),
            axis=1,
        )
        user_text_file = osp.join(self.rag_dir, "user_text.txt")
        with open(user_text_file, "w") as f:
            for text in text_attr:
                f.write(text + "\n")

        # load movie table
        movie_df = self.load_raw_df("movies.csv")
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
                f"Certificate code is: {row['Certificate']}, " if pd.notna(row['Certificate']) else "Certificate code is not known, "
                f"Plot is: {row['Plot']}, " if pd.notna(row['Plot']) else "Plot is not known, "
                f"Url is: {row['Url']}."
            ),
            axis=1,
        )
        movie_text_file = osp.join(self.rag_dir, "movie_text.txt")
        with open(movie_text_file, "w") as f:
            for text in text_attr:
                f.write(text + "\n")

        shutil.copy2(
            osp.join(self.raw_dir, "masks.pt"),
            osp.join(self.rag_dir, "masks.pt"),
        )


tacm = TACM12K()
tacm.build_tag()

tlf2k = TLF2K()
tlf2k.build_tag()

tml1m = TML1M()
tml1m.build_tag()
