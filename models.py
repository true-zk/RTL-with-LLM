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


class ImportanceEncoder(torch.nn.Module):
    def __init__(
        self,
        num_labels: int,
        embed_dim: int,
        input_size: int = 5,
        weight: List[int] = [1, 1, 1, 1, 1],
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_labels, embed_dim)
        assert input_size == len(weight)
        self.input_size = input_size
        self._weight = torch.tensor(weight, dtype=torch.float32)
        self.register_buffer("weight", self._weight)  # register as buffer

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (B, 5) -> (B, 5, embed_dim)
        x = x * self.weight.view(1, self.input_size, 1)
        x = x.view(-1, self.input_size * x.size(2))  # (B, 5, embed_dim) -> (B, 5 * embed_dim)
        return x


class MultiTableBridge(torch.nn.Module):
    def __init__(
        self,
        target_table: str,
        lin_input_dim_dict: Dict[str, int],
        graph_dim: int,
        table_encoder_dict: Dict[str, TableEncoder],
        graph_encoder: GraphEncoder,
        *,
        dropout: float = 0.5,
        llm_vec_encoder: Optional[Type[torch.nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.target_table = target_table
        self.table_encoder_dict = torch.nn.ModuleDict(table_encoder_dict)
        self.lin_dict = torch.nn.ModuleDict()
        for table_name, in_dim in lin_input_dim_dict.items():
            mlp = torch.nn.Sequential(
                torch.nn.Linear(in_dim, graph_dim),
                torch.nn.LayerNorm(graph_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            )
            self.lin_dict[table_name] = mlp
        self.graph_encoder = graph_encoder
        self.dropout = dropout

        # for llm enhence
        if llm_vec_encoder is not None:
            self.llm_vec_encoder = llm_vec_encoder

    def forward(
        self,
        table_dict: Dict[str, TableData],
        non_table_dict: Optional[Dict[str, Tensor]],
        edge_index: Tensor,
        llm_vec: Optional[Tensor] = None,
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

        # llm enhence embedding
        if hasattr(self, "llm_vec_encoder") and llm_vec is not None:
            llm_vec = self.llm_vec_encoder(llm_vec)
            x_dict[self.target_table] = torch.concat([x_dict[self.target_table], llm_vec], dim=1)

        x_dict = {
            table_name: self.lin_dict[table_name](x)
            for table_name, x in x_dict.items()
        }


        x = torch.concat(
            [i for i in x_dict.values() if i is not None], dim=0
        )

        node_feats = self.graph_encoder(x, edge_index)
        node_feats = F.dropout(node_feats, p=self.dropout, training=self.training)
        # target table always at first
        return node_feats[: target_l, :]


class FuseII(torch.nn.Module):
    def __init__(
        self,
        target_table: str,
        lin_input_dim_dict: Dict[str, int],
        graph_dim: int,
        table_encoder_dict: Dict[str, TableEncoder],
        graph_encoder: GraphEncoder,
        *,
        dropout: float = 0.5,
        bridge_out_dim: int = 0,
        llm_encoder_out_dim: int = 0,
        hidden_dim: int = 0,
        output_dim: int = 0,
        llm_vec_encoder: Optional[Type[torch.nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.multi_table_bridge = MultiTableBridge(
            target_table=target_table,
            lin_input_dim_dict=lin_input_dim_dict,
            graph_dim=graph_dim,
            table_encoder_dict=table_encoder_dict,
            graph_encoder=graph_encoder,
            dropout=dropout,
        )
        self.llm_vec_encoder = llm_vec_encoder

        self.lin_bridge = torch.nn.Linear(
            in_features=bridge_out_dim,
            out_features=hidden_dim,
        )
        self.lin_llm_embed = torch.nn.Linear(
            in_features=llm_encoder_out_dim,
            out_features=hidden_dim,
        )
        self.p = torch.nn.Parameter(torch.ones(1))
        self.lin_out = torch.nn.Linear(
            in_features=hidden_dim,
            out_features=output_dim,
        )

        self.dropout = dropout

    @property
    def factor(self):
        return torch.sigmoid(self.p)

    def forward(
        self,
        table_dict: Dict[str, TableData],
        non_table_dict: Optional[Dict[str, Tensor]],
        edge_index: Tensor,
        llm_vec: Tensor,
    ):
        bridge_out = self.multi_table_bridge(
            table_dict=table_dict,
            non_table_dict=non_table_dict,
            edge_index=edge_index
        )
        x_1 = F.relu(self.lin_bridge(bridge_out))

        llm_embed_out = self.llm_vec_encoder(llm_vec)
        x_2 = F.relu(self.lin_llm_embed(llm_embed_out))

        x = self.factor * x_1 + (1 - self.factor) * x_2 + x_2
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_out(x)
        return x

        # x_1 = F.dropout(x_1, p=self.dropout, training=self.training)
        # pred_1 = self.lin_out(x_1)

        # x_2 = llm_vec[:, 0]
        # x_2 = x_2.to(torch.int64)

        # pred_2 = F.one_hot(x_2, num_classes=15)
        # pred_2 = pred_2[:, :14].to(torch.float32)
        # out = pred_2
        # return out + self.factor * 0.001
        # x = self.factor * x_1 + (1 - self.factor) * x_2


###############################################################
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
