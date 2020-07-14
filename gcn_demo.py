import os.path as osp
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

# dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
# dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# data = dataset[0]

# add my dataset process code
def dataset_process():
    content_path = './cora/cora.content'
    cite_path = './cora/cora.cites'

    with open(content_path, "r") as f:
        contents = f.readlines()
    with open(cite_path, "r") as f:
        cites = f.readlines()

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

    x = torch.from_numpy(np.array(feature_list, dtype=np.float32))
    edge_index = torch.from_numpy(cites)
    y = torch.from_numpy(y)
    data = Data(x=x, edge_index=edge_index, y=y)

    # 分割数据集
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[:data.num_nodes - 1000] = 1  # 1708 train
    data.val_mask = None  # 0 valid
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[data.num_nodes - 500:] = 1  # 500 test
    data.num_classes = len(label_dict)

    print("{}Data Info{}".format("*" * 20, "*" * 20))
    print("==> Is undirected graph : {}".format(data.is_undirected()))
    print("==> Number of edges : {}/2={}".format(data.num_edges, int(data.num_edges / 2)))
    print("==> Number of nodes : {}".format(data.num_nodes))
    print("==> Node feature dim : {}".format(data.num_features))
    print("==> Number of training nodes : {}".format(data.train_mask.sum().item()))
    print("==> Number of testing nodes : {}".format(data.test_mask.sum().item()))
    print("==> Number of classes : {}".format(data.num_classes))

    print(f"{'-'*30} Label Info {'-'*30}")
    print(("\n{}"*7).format(*[(i,j) for j,i in label_dict.items()]))

    inds, nums = np.unique(y[data.train_mask].numpy(), return_counts=True)
    plt.figure(1)
    plt.subplot(121)
    plt.bar(x=inds, height=nums, width=0.8, bottom=0, align='center')
    plt.xticks(ticks=range(data.num_classes))
    plt.xlabel(xlabel="Label Index")
    plt.ylabel(ylabel="Sample Num")
    plt.ylim((0, 600))
    plt.title(label="Train Data Statics")
    inds, nums = np.unique(y[data.test_mask].numpy(), return_counts=True)
    plt.subplot(122)
    plt.bar(x=inds, height=nums, width=0.8, bottom=0, align='center')
    plt.xticks(ticks=range(data.num_classes))
    plt.xlabel(xlabel="Label Index")
    plt.ylabel(ylabel="Sample Num")
    plt.ylim((0, 600))
    plt.title(label="Test Data Statics")
    plt.savefig("dataset_info.png")

    return data


# my dataset process function
data = dataset_process()

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, 16, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, data.num_classes, cached=True,
                             normalize=not args.use_gdc)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask].long()).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 21):
    train()
    train_acc, test_acc = test()
    log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, test_acc))