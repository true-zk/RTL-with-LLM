from torch_geometric.data import HeteroData

from rllm.data import HeteroGraphData

from llm import llm
from bge_model import embed_model, tokenizer

from dataprocess import TACM12K


class BaseSTAPE:

    def __init__(self):
        self.llm = llm
        self.embed_model = embed_model
        self.tokenizer = tokenizer

    def pyg_graph(self):
        raise NotImplementedError

    def sampler(self):
        raise NotImplementedError


class TACM12K_STAPE(BaseSTAPE):
    def __init__(self):
        super(TACM12K_STAPE, self).__init__()
        (
            self.hgraph,
            self.author_text,
            self.paper_text,
            self.y,
            self.masks
        ) = TACM12K.load_all()

    def pyg_graph(self):
        # Convert HeteroGraphData to PyG Data
        pyg_graph = HeteroData()
        for edge_type in self.hgraph.edge_types:
            edge_d = self.hgraph[edge_type].to_dict()
            for k, v in edge_d.items():
                pyg_graph[edge_type][k] = v
        print(pyg_graph)



a = TACM12K_STAPE()
a.pyg_graph()
