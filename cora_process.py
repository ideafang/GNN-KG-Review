import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, ChebConv, MessagePassing, SplineConv
from torch_geometric.utils import add_self_loops, degree

content_path = './cora/cora.content'
cite_path = './cora/cora.cites'

with open(content_path, "r") as f:
    contents = f.readlines()
with open(cite_path, "r") as f:
    cites = f.readlines()

# contents, cites are lists
# print(np.array(contents).shape) # (2708,)
# print(np.array(cites).shape) # (5429,)
# print(contents[0]) # \t划分数据

# contents数据切分 -> <paper> + <feature> + <label>
contents = np.array([np.array(line.strip().split("\t")) for line in contents])
# print(contents.shape) # (2708, 1435)
paper_list, feature_list, label_list = np.split(contents, [1, -1], axis=1)
paper_list, label_list = np.squeeze(paper_list), np.squeeze(label_list)

# paper -> dict
paper_dict = dict([(key, val) for val, key in enumerate(paper_list)])
# print(paper_dict[31336]) # '31336': 0

# label -> dict
labels = list(set(label_list))
label_dict = dict([(key, val) for val, key in enumerate(labels)])
# print(label_dict['Rule_learning']) # 'Rule_Learning': 0

# cites数据整理
cites = [line.strip().split("\t") for line in cites]
# 将cites中引用关系的paperID转换为paper_dict字典序，最后转置矩阵是为了满足PyG输入中edge_index的要求
# cite_id[0]为被引用文献的paper_id, cite_id[1]为引用文献的paper_id
cites = np.array([[paper_dict[cite_id[0]], paper_dict[cite_id[1]]] for cite_id in cites], np.int64).T
# cites.shape = (2, 5429)
cites = np.concatenate((cites, cites[::-1, :]), axis=1)
# cites.shape = (2, 5429*2)

# y构建
y = np.array([label_dict[i] for i in label_list])

# Input
node_num = len(paper_list)         # 节点个数
feat_dim = feature_list.shape[1]   # 特征维度
stat_dim = 32                      # 状态维度
num_class = len(labels)            # 节点种类数

x = torch.from_numpy(np.array(feature_list, dtype=np.float32))
edge_index = torch.from_numpy(cites)
y = torch.from_numpy(y)
data = Data(x = x, edge_index = edge_index, y = y)

# 分割数据集
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
data.train_mask[:data.num_nodes - 1000] = 1  #1708 train
data.val_mask = None                        #0 valid
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
data.test_mask[data.num_nodes - 500:] = 1  #500 test
data.num_classes = len(label_dict)

print("{}Data Info{}".format("*"*20, "*"*20))
print("==> Is undirected graph : {}".format(data.is_undirected()))
print("==> Number of edges : {}/2={}".format(data.num_edges, int(data.num_edges/2)))
print("==> Number of nodes : {}".format(data.num_nodes))
print("==> Node feature dim : {}".format(data.num_node_features))
print("==> Number of training nodes : {}".format(data.train_mask.sum().item()))
print("==> Number of testing nodes : {}".format(data.test_mask.sum().item()))
print("==> Number of classes : {}".format(data.num_classes))

# 构建GCN 使用官方教程的例子
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)  # (N, in_channels) -> (N, out_channels)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]
        # edge_index has shape [2, E]

        # Step 3: Normalize node features.
        row, col = edge_index  # [E,], [E,]
        deg = degree(row, size[0], dtype=x_j.dtype)  # [N, ]
        deg_inv_sqrt = deg.pow(-0.5)   # [N, ]
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # [E, ]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

# 构建模型，使用两层GCN
class Net(torch.nn.Module):
    def __init__(self, feat_dim, num_class):
        super(Net, self).__init__()
        self.conv1 = GCNConv(feat_dim, 16)
        self.conv2 = GCNConv(16, num_class)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(feat_dim=data.num_node_features, num_class=7).to(device)         # Initialize model
data = data.to(device)                                                       
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) # Initialize optimizer and training params

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 201):
    train()
    log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, *test()))