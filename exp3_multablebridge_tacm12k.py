# The BRIDGE method from the "rLLM: Relational Table Learning with LLMs" paper.
# ArXiv: https://arxiv.org/abs/2407.20157

# Datasets  TACM12K
# Acc       0.293
# MyAcc     0.465

import time
import argparse
import sys

import torch
import torch.nn.functional as F

sys.path.append("./")
sys.path.append("../")
from rllm.data import GraphData
from rllm.transforms.graph_transforms import GCNTransform
from rllm.transforms.table_transforms import TabTransformerTransform
from rllm.nn.conv.graph_conv import GCNConv, GATConv
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.models import TableEncoder, GraphEncoder

from models import MultiTableBridge
from rllm_loader import load_dataset

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data, g = load_dataset(dataset_name='tacm12k', load_graph=True)
paper_embeddings = data['paper_embeddings']
papers_table = data['papers']


target_table = papers_table.to(device)
y = papers_table.y.long().to(device)
paper_embeddings = paper_embeddings.to(device)

edge_index1 = g['paper', 'cites', 'paper'].edge_index
edge_index2 = g['paper', 'written_by', 'author'].edge_index
# paper | author
edge_index2[1] += len(papers_table)
adj = torch.concat(
    [
        edge_index1,
        edge_index2,
        edge_index2.flip(0)
    ],
    dim=-1
)
num_nodes = len(papers_table) + len(data['authors'])
adj = torch.sparse_coo_tensor(
    indices=adj,
    values=torch.ones(adj.size(1)),
    size=(num_nodes, num_nodes),
).to(device)
g = GraphData(adj=adj)

####################################################################
paper_emb_size = paper_embeddings.size(1)
paper_table_dim = 16
table_transform = TabTransformerTransform(
    out_dim=paper_table_dim, metadata=target_table.metadata
)
target_table = table_transform(data=target_table)

author_emb_size = data['author_embeddings'].size(1)
author_table_dim = 16
author_table_transform = TabTransformerTransform(
    out_dim=author_table_dim, metadata=data['authors'].metadata
)
data['authors'] = author_table_transform(data=data['authors'])

graph_transform = GCNTransform()
adj = graph_transform(data=g).adj
adj = adj.to(device)

####################################################################
table_dict = {
    'paper': target_table.to(device),
    'author': data['authors'].to(device),
}
# non_table_dict = {
#     'paper': paper_embeddings[len(target_table):, :],
# }
non_table_dict = {
    'paper': paper_embeddings[:, :],
    'author': data['author_embeddings'].to(device),
}
table_embed_dim_dict = {
    'paper': paper_emb_size + paper_table_dim,
    'author': author_emb_size + author_table_dim,
}
node_embed = 64
####################################################################

# Split data
train_mask, val_mask, test_mask = (
    target_table.train_mask,
    target_table.val_mask,
    target_table.test_mask,
)

# Set up model and optimizer
t_encoder_paper = TableEncoder(
    in_dim=paper_table_dim,
    out_dim=paper_table_dim,
    table_conv=TabTransformerConv,
    metadata=target_table.metadata,
)
t_encoder_author = TableEncoder(
    in_dim=author_table_dim,
    out_dim=author_table_dim,
    table_conv=TabTransformerConv,
    metadata=data['authors'].metadata,
)
g_encoder = GraphEncoder(
    in_dim=node_embed,
    out_dim=target_table.num_classes,
    graph_conv=GCNConv,
)
model = MultiTableBridge(
    target_table='paper',
    lin_input_dim_dict=table_embed_dim_dict,
    graph_dim=node_embed,
    table_encoder_dict={
        'paper': t_encoder_paper,
        'author': t_encoder_author,
    },
    graph_encoder=g_encoder,
    dropout=0.4,
).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    # weight_decay=args.wd,
)
####################################################################

def train() -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(
        table_dict=table_dict,
        non_table_dict=non_table_dict,
        edge_index=adj,
    )
    loss = F.cross_entropy(logits[train_mask].squeeze(), y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test():
    model.eval()
    logits = model(
        table_dict=table_dict,
        non_table_dict=non_table_dict,
        edge_index=adj,
    )
    preds = logits.argmax(dim=1)

    accs = []
    for mask in [train_mask, val_mask, test_mask]:
        correct = float(preds[mask].eq(y[mask]).sum().item())
        accs.append(correct / int(mask.sum()))
    return accs


def main():
    start_time = time.time()
    best_val_acc = best_test_acc = 0
    # val_acc_list = []
    # test_acc_list = []
    # train_acc_list = []
    # loss_list = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train()
        train_acc, val_acc, test_acc = test()
        # loss_list.append(train_loss)

        # val_acc_list.append(val_acc)
        # train_acc_list.append(train_acc)
        # test_acc_list.append(test_acc)
        print(
            f"Epoch: [{epoch}/{args.epochs}]"
            f"Loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
            f"val_acc: {val_acc:.4f} test_acc: {test_acc:.4f} "
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

    print(f"Total Time: {time.time() - start_time:.4f}s")
    print(
        "MultiTableBRIDGE result: "
        f"Best Val acc: {best_val_acc:.4f}, "
        f"Best Test acc: {best_test_acc:.4f}"
    )

    #     with open('./res_exp3.pt', 'wb') as f:
    #         pickle.dump(
    #             {
    #                 'loss': loss_list,
    #                 'train_acc': train_acc_list,
    #                 'val_acc': val_acc_list,
    #                 'test_acc': test_acc_list
    #             },
    #             f
    #         )

    return best_test_acc


main()


#################################################################

# epochs_l = np.arange(1, 101)


# plt.figure(figsize=(10, 5))
# plt.plot(epochs_l, loss_list, label='Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Loss vs Epochs')
# plt.legend()
# # plt.savefig('./mbridge_loss_plot.png')
# plt.show()


# best_val_idx = np.argmax(val_acc)
# plt.figure(figsize=(10, 5))
# plt.plot(epochs_l, train_acc_list, label='Train Acc', color='#1f77b4')
# plt.plot(epochs_l, val_acc_list, label='Val Accu', color='#1f77b4')
# plt.plot(epochs_l, test_acc_list, label='Test Accu', color='#1f77b4')
# plt.text(epochs_l[best_val_idx], val_acc[best_val_idx] + 0.005,
#         f'val:{val_acc[best_val_idx]:.4f}, test:{test_acc[best_val_idx]:.4f}',
#         ha='center', va='bottom', fontsize=9, color='red')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
# plt.savefig('./mbridge_acc_plot.png')
