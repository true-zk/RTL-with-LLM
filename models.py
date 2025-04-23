from typing import List, Dict, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, Linear, SAGEConv, HANConv
from torch_geometric.nn import GCNConv as pygGCNConv

from rllm.nn.conv.graph_conv import GCNConv


class PygGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        self.conv1 = pygGCNConv(in_dim, hidden_dim)
        self.conv2 = pygGCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class TrivialGNN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=True))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=True))
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        out = self.lin(x)
        return out


class HeteroSAGE(torch.nn.Module):
    def __init__(
        self,
        target_node_type: str,
        metadata: List,
        inchannels: Dict,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        *,
        virtual_node_type: Optional[str] = None,
        num_virtual_nodes: Optional[int] = None,
    ):
        super().__init__()

        self.target_node_type = target_node_type

        self.mlp = torch.nn.ModuleDict()
        for node_type, inchannel in inchannels.items():
            self.mlp[node_type] = torch.nn.Linear(inchannel, 128)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # Assign each edge type a individual conv
            conv = HeteroConv(
                convs={
                    edge_type: SAGEConv(
                        in_channels=(-1, -1),
                        out_channels=hidden_channels,
                        normalize=True,
                    )
                    for edge_type in metadata[1]
                }
            )
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

        # set virtual nodes feature
        self.virtual_node_type = virtual_node_type
        if virtual_node_type is not None:
            print("Add virtual node feature")
            assert num_virtual_nodes > 0
            self.virtual_node = torch.nn.Embedding(
                num_embeddings=num_virtual_nodes,
                embedding_dim=hidden_channels
            )

    def forward(self, x_dict: Dict, edge_index_dict: Dict):
        if self.virtual_node_type is not None:
            x_dict[self.virtual_node_type] = self.virtual_node.weight

        for node_type, x in x_dict.items():
            x_dict[node_type] = self.mlp[node_type](x)

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        out = self.lin(x_dict[self.target_node_type])
        # out = F.dropout(out, p=0.5, training=self.training)
        return out


class HAN(torch.nn.Module):
    def __init__(
        self,
        metadata: List,
        target_node_type: str,
        out_channels: int,
        hidden_channels: int = 128,
        heads: int = 8,
        *,
        virtual_node_type: Optional[str] = None,
        num_virtual_nodes: Optional[int] = None,
    ):
        super().__init__()
        self.target_node_type = target_node_type
        self.conv = HANConv(
            in_channels=-1,
            out_channels=hidden_channels,
            metadata=metadata,
            heads=heads,
            dropout=0.5,
        )
        self.lin = Linear(hidden_channels, out_channels)
        # set virtual nodes feature
        self.virtual_node_type = virtual_node_type
        if virtual_node_type is not None:
            print("Add virtual node feature")
            assert num_virtual_nodes > 0
            self.virtual_node = torch.nn.Embedding(
                num_embeddings=num_virtual_nodes,
                embedding_dim=hidden_channels
            )

    def forward(self, x_dict: Dict, edge_index_dict: Dict):
        if self.virtual_node_type is not None:
            x_dict[self.virtual_node_type] = self.virtual_node.weight
        x_dict = self.conv(x_dict, edge_index_dict)
        # x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return self.lin(x_dict[self.target_node_type])


from typing import Type, Union, Tuple
from rllm.data import TableData, HeteroGraphData
from rllm.nn.models import TableEncoder, GraphEncoder
from rllm.nn.conv.graph_conv import HANConv


class MultiTableBridge(torch.nn.Module):
    def __init__(
        self,
        target_table: str,
        table_dim: Dict[str, int],
        graph_dim: int,
        table_encoder_dict: Dict[str, TableEncoder],
        graph_encoder: GraphEncoder,
    ) -> None:
        super().__init__()
        self.target_table = target_table
        self.table_encoder_dict = torch.nn.ModuleDict(table_encoder_dict)
        self.lin_dict = torch.nn.ModuleDict()
        for table_name, in_dim in table_dim.items():
            self.lin_dict[table_name] = torch.nn.Linear(in_dim, graph_dim)
        self.graph_encoder = graph_encoder


    def forward(
        self,
        table_dict: Dict[str, TableData],
        non_table_dict: Optional[Dict[str, Tensor]],
        edge_index: Tensor,
    ) -> Tensor:
        target_l = len(table_dict[self.target_table])
        x_dict = {
            table_name: self.table_encoder_dict[table_name](table)
            for table_name, table in table_dict.items()
        }

        if non_table_dict is not None:
            for table_name, embed in non_table_dict.items():
                if embed.numel() != 0:
                    x_dict[table_name] = torch.concat(
                        [x_dict[table_name], embed], dim=1
                    )

        x_dict = {
            table_name: self.lin_dict[table_name](x)
            for table_name, x in x_dict.items()
        }

        x = torch.concat(
            [i for i in x_dict.values() if i is not None], dim=0
        )

        node_feats = self.graph_encoder(x, edge_index)
        # target table always at first
        return node_feats[: target_l, :]


class HGraphEncoder(torch.nn.Module):
    def __init__(
        self,
        in_dim: Union[int, Dict[str, int]],
        hidden_dim: int,
        out_dim: int,
        metadata: Tuple[List[str], List[Tuple[str, str]]],
        num_heads: int = 1,
        dropout: float = 0.5,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers - 1):
            self.convs.append(HANConv(in_dim=in_dim, out_dim=hidden_dim, metadata=metadata, num_heads=num_heads))
        self.convs.append(HANConv(in_dim=hidden_dim, out_dim=out_dim, metadata=metadata, num_heads=num_heads))

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Tensor],
    ) -> Tensor:
        for conv in self.convs[:-1]:
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}
        x_dict = self.convs[-1](x_dict, edge_index_dict)
        return x_dict


class HGraphNN(torch.nn.Module):
    def __init__(
        self,
        target_node_type: str,
        in_dim: Union[int, Dict[str, int]],
        hidden_dim: int,
        out_dim: int,
        metadata: Tuple[List[str], List[Tuple[str, str]]],
        num_heads: int = 1,
        dropout: float = 0.5,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.target_node_type = target_node_type
        self.pre_lin_dict = torch.nn.ModuleDict()
        for node_type, inchannel in in_dim.items():
            self.pre_lin_dict[node_type] = torch.nn.Linear(inchannel, hidden_dim)
        self.HGraphEncoder = HGraphEncoder(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            metadata=metadata,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.readout = torch.nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Tensor],
    ) -> Tensor:
        x_dict = {k: F.relu(self.pre_lin_dict[k](v)) for k, v in x_dict.items()}
        x_dict = self.HGraphEncoder(x_dict, edge_index_dict)
        out = self.readout(x_dict[self.target_node_type])
        # out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class HBRIDGE(torch.nn.Module):

    def __init__(
        self,
        table_encoder_dict: Dict[str, TableEncoder],
        graph_encoder: HGraphEncoder,
    ) -> None:
        super().__init__()
        self.table_encoder_dict = table_encoder_dict
        self.graph_encoder = graph_encoder

    def forward(
        self,
        table_dict: Dict[str, TableData],
        hetero_graph: HeteroGraphData,
        non_table_dict: Dict[str, Tensor] = None,
    ) -> Tensor:
        # Tables -- Table convs -- > table_out
        table_out = {}
        for table_name, table in table_dict.items():
            table_out[table_name] = self.table_encoder_dict[table_name](table)

        # Concat table_out with non_table_dict
        if non_table_dict is not None:
            for table_name, embed in non_table_dict.items():
                table_out[table_name] = torch.concat(
                    [table_out[table_name], embed], dim=1
                )

        graph_out = self.graph_encoder(table_out, hetero_graph)

        out = torch.cat([*table_out.values(), graph_out], dim=1)
        return out

