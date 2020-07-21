import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

content_path = "./cora/cora.content"
cite_path = "./cora/cora.cites"

# 读取文本内容
with open(content_path, "r") as fp:
    contents = fp.readlines()
with open(cite_path, "r") as fp:
    cites = fp.readlines()

contents = np.array([np.array(l.strip().split("\t")) for l in contents])
paper_list, feat_list, label_list = np.split(contents, [1,-1], axis=1)
paper_list, label_list = np.squeeze(paper_list), np.squeeze(label_list)
# Paper -> Index dict
paper_dict = dict([(key, val) for val, key in enumerate(paper_list)])
# Label -> Index 字典
labels = list(set(label_list))
label_dict = dict([(key, val) for val, key in enumerate(labels)])
# Edge_index
cites = [i.strip().split("\t") for i in cites]
cites = np.array([[paper_dict[i[0]], paper_dict[i[1]]] for i in cites],
                 np.int64).T   # (2, edge)
cites = np.concatenate((cites, cites[::-1, :]), axis=1)  # (2, 2*edge) or (2, E)
# Degree
_, degree_list = np.unique(cites[0,:], return_counts=True)

# Input
node_num = len(paper_list)
feat_dim = feat_list.shape[1]
stat_dim = 32
num_class = len(labels)
T = 2
feat_Matrix = torch.Tensor(feat_list.astype(np.float32))
X_Node, X_Neis = np.split(cites, 2, axis=0)
X_Node, X_Neis = torch.from_numpy(np.squeeze(X_Node)), \
                 torch.from_numpy(np.squeeze(X_Neis))
dg_list = degree_list[X_Node]
label_list = np.array([label_dict[i] for i in label_list])
label_list = torch.from_numpy(label_list)

print("{}Data Process Info{}".format("*"*20, "*"*20))
print("==> Number of node : {}".format(node_num))
print("==> Number of edges : {}/2={}".format(cites.shape[1], int(cites.shape[1]/2)))
print("==> Number of classes : {}".format(num_class))
print("==> Dimension of node features : {}".format(feat_dim))
print("==> Dimension of node state : {}".format(stat_dim))
print("==> T : {}".format(T))
print("==> Shape of feat_Matrix : {}".format(feat_Matrix.shape))
print("==> Shape of X_Node : {}".format(X_Node.shape))
print("==> Shape of X_Neis : {}".format(X_Neis.shape))
print("==> Length of dg_list : {}".format(len(dg_list)))

# Split dataset
train_mask = torch.zeros(node_num, dtype=torch.uint8)
train_mask[:node_num - 1000] = 1                  # 1700左右training
val_mask = None                                    # 0valid
test_mask = torch.zeros(node_num, dtype=torch.uint8)
test_mask[node_num - 500:] = 1                    # 500test
x = feat_Matrix
edge_index = torch.from_numpy(cites)

class AggrSum(nn.Module):
    def __init__(self, node_num):
        super(AggrSum, self).__init__()
        self.V = node_num

    def forward(self, H, X_node):
        mask = torch.stack([X_node] * self.V, 0)
        mask = mask.float() - torch.unsqueeze(torch.range(0, self.V-1).float(), 1)
        mask = (mask == 0).float()
        return torch.mm(mask, H)

class GCNConv(nn.Module):
    def __init__(self, in_channel, out_channel, node_num):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.aggregation = AggrSum(node_num)

    def forward(self, x, edge_index):
        edge_index = self.addSelfConnect(edge_index, x.shape[0])

        x = self.linear(x)

        row, col = edge_index
        deg = self.calDegree(row, x.shape[0]).float()
        deg_sqrt = deg.pow(-0.5)
        norm = deg_sqrt[row] * deg_sqrt[col]

        tar_matrix = torch.index_select(x, dim=0, index=col)
        tar_matrix = norm.view(-1, 1) * tar_matrix

        aggr = self.aggregation(tar_matrix, row)
        return aggr

    def addSelfConnect(self, edge_index, num_nodes):
        selfconn = torch.stack([torch.range(0, num_nodes-1, dtype=torch.long)]*2, dim=0).to(edge_index.device)
        return torch.cat(tensors=[edge_index, selfconn], dim=1)

    def calDegree(self, edges, num_nodes):
        ind, deg = np.unique(edges.cpu().numpy, return_counts=True)
        deg_tensor = torch.zeros((num_nodes, ), dtype=torch.long)
        deg_tensor[ind] = torch.from_numpy(deg)
        return deg_tensor.to(edges.device)

class Net(nn.Module):
    def __init__(self, feat_dim, num_class, num_node):
        super(Net, self).__init__()
        self.conv1 = GCNConv(feat_dim, 16, num_node)
        self.conv2 = GCNConv(16, num_class, num_node)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(feat_dim, num_class, node_num).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
x = x.to(device)
edge_index = edge_index.to(device)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    # Get output
    out = model(x, edge_index)

    # Get loss
    loss = F.nll_loss(out[train_mask], label_list[train_mask])
    _, pred = out.max(dim=1)

    # Get predictions and calculate training accuracy
    correct = float(pred[train_mask].eq(label_list[train_mask]).sum().item())
    acc = correct / train_mask.sum().item()
    print('[Epoch {}/200] Loss {:.4f}, train acc {:.4f}'.format(epoch, loss.cpu().detach().data.item(), acc))

    # Backward
    loss.backward()
    optimizer.step()

    # Evaluation on test data every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        _, pred = model(x, edge_index).max(dim=1)
        correct = float(pred[test_mask].eq(label_list[test_mask]).sum().item())
        acc = correct / test_mask.sum().item()
        print('Accuracy: {:.4f}'.format(acc))