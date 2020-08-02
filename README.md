# GNN调研

## 目录：

- [GNN概述及应用](#GNN概述及应用)

  - [GNN简介](#GNN简介)

  - [GNN起源](#GNN起源)
  - [GNN和传统神经网络的区别](#GNN和传统神经网络的区别)
  - [GNN分类](#GNN分类)

- [GNN实战](#GNN实战)

  - [PyG框架安装](#PyG框架安装)
  - [PyG框架测试](#PyG框架测试)
  - [PyG输入数据格式](#PyG输入数据格式)
  - [Cora数据集介绍](#Cora数据集介绍)
  - [Cora数据预处理](#Cora数据预处理)
  - [基于GCN的节点分类问题demo](#基于GCN的节点分类问题demo)

- [Pytorch学习](#Pytorch学习)

- [Graph图存储](#Graph图存储)

  - [Neo4j介绍](#Neo4j介绍)
  - [Cora2neo](#Cora2neo)

## 任务：

弄清GNN的输入格式、关系型、CSV、JSON等，用Demo说明

## GNN概述及应用

### awosome information resource：

- 清华大学NLP组：GNN介绍及必读文章https://github.com/thunlp/GNNPapers
- 知乎：GNN阅读合集https://zhuanlan.zhihu.com/c_1139920494823342080



### 《Graph Neural Networks: A Review of Methods and Applications》

#### GNN简介

> Graph neural networks (GNNs) are connectionist models that capture the dependence of graphs via message passing between the nodes of graphs

GNN是一种连接模型，通过学习图节点之间的连接关系来捕获图的依赖特征。

> Recently, researches of analyzing graphs with machine learning have been receiving more and more attention because of the great expressive power of graphs.

最近，由于图具有优秀的表达能力，因此在机器学习中对图的研究获得了越来越多的关注。

> As a unique non-Euclidean data structure for machine learning, graph analysis focuses on node classification, link prediction, and clustering.

图作为机器学习中一种独特的非欧几里得数据结构，对图的分析主要专注于节点分类、关系预测、聚类等方面。

#### GNN起源

GNN的起源：CNN和图嵌入（graph embedding）

> As we are going deeper into CNNs and graphs, we found the keys of CNNs: local connection, shared weights and the use of multi-layer.

CNN的核心特点：局部连接、权重共享、多层叠加。（这些在图中也可以使用，因为图是最经典的局部连接结构，但图属于非欧几里得数据，CNN的卷积和池化操作很难迁移到图中，因此诞生了GNN）

![image-20200709232609888](picture/image-20200709232609888-1594393772925.png)

> The other motivation comes from graph embedding [11]– [15], which learns to represent graph nodes, edges or subgraphs in low-dimensional vectors.

所谓嵌入，就是对图的节点、边、或者子图学习得到一个低维的向量表示。

基于表征学习的图嵌入方法缺陷：

1. 节点编码中权重未共享，导致权重参数随着节点增多而线性增大
2. 直接嵌入方法缺乏泛化能力，无法处理动态图以及泛化到新的图

#### GNN和传统神经网络的区别

由于传统的神经网络都需要节点之间按照标准的顺序排列，因此传统神经网络CNN、RNN都不能适当地处理图结构输入。

> GNNs propagate on each node respectively, ignoring the input order of nodes. In other words, the output of GNNs is invariant for the input order of nodes.

GNN采用在每个节点上分别传播（propagate）的方式进行学习，由此忽略了节点的顺序。即GNN的输出会随着输入的不同而不同。

> Generally, GNNs update the hidden state of nodes by a weighted sum of the states of their neighborhood.

通常，GNN通过对其相邻节点的加权求和来更新节点的隐藏状态。

#### GNN分类

- 图卷积网络（Graph Convolutional Networks, GCN）和图注意力网络（Graph Attention Networks, GAN）。因为两者都涉及到传播步骤（propagation step）
- 图空域网络（Graph spatial-temporal networks）。因为该模型常用在动态图上。
- 图自编码器（Graph AutoEncoder），因为该模型通常使用无监督学习方式（unsupervised）
- 图生成网络（Graph Generative Networks）,因为是生成式网络。

## GNN实战

### PyG框架安装

#### Pytorch安装

使用anaconda+cuda10.2环境

https://pytorch.org/get-started/locally/ 在网站中选择配置参数，获取安装命令

![image-20200710223712332](picture/image-20200710223712332-1594393778389.png)

```shell
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

最新版本为pytorch=1.5.1

#### PyG模块安装

（需科学上网，不然可能因为下载包不全而报错）

```shell
pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-geometric
```

### PyG框架测试

PyG项目GitHub地址：https://github.com/rusty1s/pytorch_geometric

测试用例：

```shell
$ cd examples
$ python gcn.py
```

测试结果：

```shell
(gnn) F:\OneDrive\研一\GNN\pytorch_geometric-master>cd examples

(gnn) F:\OneDrive\研一\GNN\pytorch_geometric-master\examples>python gcn.py
Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.x
Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.tx
Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.allx
Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.y
Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.ty
Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.ally
Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.graph
Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.test.index
Processing...
Done!
Epoch: 001, Train: 0.2786, Val: 0.2400, Test: 0.2490
Epoch: 002, Train: 0.5857, Val: 0.3280, Test: 0.3390
Epoch: 003, Train: 0.6571, Val: 0.3640, Test: 0.3680
......
Epoch: 197, Train: 0.9929, Val: 0.8000, Test: 0.8210
Epoch: 198, Train: 0.9929, Val: 0.8000, Test: 0.8210
Epoch: 199, Train: 0.9929, Val: 0.8000, Test: 0.8210
Epoch: 200, Train: 0.9929, Val: 0.8000, Test: 0.8210

(gnn) F:\OneDrive\研一\GNN\pytorch_geometric-master\examples>
```

### PyG输入数据格式

PyG官方文档：https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

在PyG中，图（graph）被用来构建节点（nodes）之间的关系（relations）。每个graph都被定义为`torch_geometric.data.Data`，默认具有以下属性**（非必须）**：

- `data.x`：节点特征矩阵，shape为`[num_nodes, num_node_features]`

- `data.edge_index`：COO格式的图连接（Graph connectivity）数据，shape为`[2, num_edges]`，type为`torch.long`

- `data.edge_attr`：边特征矩阵，shape为`[num_edges, num_edges_features]`

- `data.y`：训练目标（Target to train against），shape不固定。例如：对于node-level的任务，shape为`[num_nodes, *]`，对于graph-level的任务，shape为`[1, *]`。

  > Applications to a graphical domain can generally be divided into two broad classes, called **graph-focused** and **node-focused**. ——《The Graph Neural Network Model》

- `data.pos`：节点的位置矩阵（position matrix），shape为`[num_nodes, num_dimensions]`

实例：

![image-20200711201832839](picture/image-20200711201832839.png)

如图为一个不带权重的无向图，有三个顶点0、1、2，以及两条无向边，每个顶点的特征维度为1即$X_1$

```python
import torch
from torch_geometric.data import Data
edge_index = torch.tensor([0, 1, 1, 2],
                          [1, 0, 2, 1], dtype=torch.long)
x = torch.tensor([-1, 0, 1], dtype=torch.float)
data = Data(x = x, edge_index = edge_index)
```

### Cora数据集介绍

Cora数据集（引文网络）是由众多机器学习领域的论文组成，是近年来图深度学习领域最常用的数据集之一，这些论文总共被分为7个类别：

- 基于案例Case based
- 生成算法Genetic Algorithms
- 神经网络Neural Networks
- 概率方法Probabilistic Methods
- 强化学习Reinforcement Leraning
- 规则学习Rule Learning
- 理论Theory

在该数据集中，每一篇论文至少引用了该数据集中的其他论文或者被其他论文所引用，总共有2708篇papers。每篇论文有1433个`word_attributes`，即每个节点的特征维度是1433.

数据集文件夹共包含两个文件：

1. `.content`：包含对每一个paper的描述，格式为：

   `<paper_id> + <word_attributes> + <class_label>`

   - `paper_id`：paper的唯一标识符
   - `word_attributes`：词汇特征，取值为0或1，表示对应词汇是否存在
   - `class_label`：论文的类别

   数据样例：

   ```shell
   31336	0	0	0	0	0	0	0	0  ...	0	0	0	0	0	0	Neural_Networks
   1061127	0	0	0	0	0	0	0	0  ...	0	0	0	1	0	0	Rule_Learning
   ```

2. `.cite`：包含数据集的引用图（citation graph），格式为：

   `<ID of cited paper> + <ID of citing paper>`

   例如有一行为`paper1 paper2`，表示`paper2`引用了`paper1`。

   数据样例：

   ```shell
   35	1033
   35	103482
   35	103515
   35	1050679
   ```

### Cora数据预处理

需要从cora原始文件中提取出相关数据信息，并转换为GNN能够识别的格式。

GNN的输入数据简单来讲需要以下数据：

- 节点node
- 节点特征矩阵node feature matrix
- 边edge
- 边的特征矩阵edge feature matrix（非必须）
- 训练目标y

对于Cora数据集来说，可以从`.content`文件中提取节点的相关数据，包括节点，节点特征矩阵，训练目标；可以从`.cite`文件中提取边的数据，在本例中没有边的特征矩阵数据。

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data


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

# 输出图信息
print("{}Data Info{}".format("*"*20, "*"*20))
print("==> Is undirected graph : {}".format(data.is_undirected()))
print("==> Number of edges : {}/2={}".format(data.num_edges, int(data.num_edges/2)))
print("==> Number of nodes : {}".format(data.num_nodes))
print("==> Node feature dim : {}".format(data.num_node_features))
print("==> Number of training nodes : {}".format(data.train_mask.sum().item()))
print("==> Number of testing nodes : {}".format(data.test_mask.sum().item()))
print("==> Number of classes : {}".format(data.num_classes))
```

最终得到的Graph信息如下：

```shell
********************Data Info********************
==> Is undirected graph : True
==> Number of edges : 10858/2=5429
==> Number of nodes : 2708
==> Node feature dim : 1433
==> Number of training nodes : 1708
==> Number of testing nodes : 500
==> Number of classes : 7
```

### 基于GCN的节点分类问题demo

训练和分类代码来自于PyG的官方实例gcn.py，对Cora数据集的预处理替换为上文中的方法。

```python
import os.path as osp
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv

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

    # 输出数据集的有关数据
    print("{}Data Info{}".format("*" * 20, "*" * 20))
    print("==> Is undirected graph : {}".format(data.is_undirected()))
    print("==> Number of edges : {}/2={}".format(data.num_edges, int(data.num_edges / 2)))
    print("==> Number of nodes : {}".format(data.num_nodes))
    print("==> Node feature dim : {}".format(data.num_features))
    print("==> Number of training nodes : {}".format(data.train_mask.sum().item()))
    print("==> Number of testing nodes : {}".format(data.test_mask.sum().item()))
    print("==> Number of classes : {}".format(data.num_classes))

    # 输出数据集各类别的条形统计图
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
```

为了缩短计算时常并便于显示，训练轮次设定为20，此外还生成了训练集和测试集中各类别节点的数量情况：

<img src="picture/dataset_info.png" alt="dataset_info" style="zoom: 80%;" />

训练结果如下：

```shell
(gnn) F:\OneDrive\研一\GNN\GNN-KG-Review>python gcn_demo.py
********************Data Info********************
==> Is undirected graph : True
==> Number of edges : 10858/2=5429
==> Number of nodes : 2708
==> Node feature dim : 1433
==> Number of training nodes : 1708
==> Number of testing nodes : 500
==> Number of classes : 7
------------------------------ Label Info ------------------------------

(0, 'Probabilistic_Methods')
(1, 'Neural_Networks')
(2, 'Rule_Learning')
(3, 'Reinforcement_Learning')
(4, 'Theory')
(5, 'Genetic_Algorithms')
(6, 'Case_Based')
Epoch: 001, Train: 0.4139, Test: 0.3660
Epoch: 002, Train: 0.4256, Test: 0.3700
Epoch: 003, Train: 0.4690, Test: 0.4020
Epoch: 004, Train: 0.5498, Test: 0.4700
Epoch: 005, Train: 0.6399, Test: 0.5440
Epoch: 006, Train: 0.6996, Test: 0.5840
Epoch: 007, Train: 0.7248, Test: 0.6260
Epoch: 008, Train: 0.7424, Test: 0.6600
Epoch: 009, Train: 0.7547, Test: 0.6660
Epoch: 010, Train: 0.7646, Test: 0.6880
Epoch: 011, Train: 0.7728, Test: 0.7000
Epoch: 012, Train: 0.7863, Test: 0.7160
Epoch: 013, Train: 0.7974, Test: 0.7280
Epoch: 014, Train: 0.8185, Test: 0.7340
Epoch: 015, Train: 0.8320, Test: 0.7420
Epoch: 016, Train: 0.8460, Test: 0.7580
Epoch: 017, Train: 0.8612, Test: 0.7820
Epoch: 018, Train: 0.8712, Test: 0.7860
Epoch: 019, Train: 0.8852, Test: 0.7960
Epoch: 020, Train: 0.8964, Test: 0.8020

(gnn) F:\OneDrive\研一\GNN\GNN-KG-Review>
```



## Pytorch学习

见pytorch文件夹中的[README](./pytorch/README.md)文档

## Graph图存储

### neo4j介绍

Neo4j是最常见的图数据库，其他还有JanusGraph，HuguGraph，TigerGraph，Gstore很多种类。其中Neo4j是最常用，也是完成度最高，上手最快的一款图数据库应用。目前大部分知识图谱均使用Neo4j作为图存储工具，因此其社区教程和QA相比其他图数据库更加完善。

### Cora2neo

在Neo4j中，采用先添加节点Node，然后添加节点之间的关系Relationship。在Python下对Neo4j读写采用的是py2neo库。

#### 添加节点

从`cora.content`中得到`paper_list`和`feature_list`之后，通过创建Node类，然后调用create方法添加节点

```python
graph.delete_all() # 清空图数据库
for i in range(paper_list.shape[0]):
    feature = dict([(str(val), key) for val, key in enumerate(feature_list[i])])
    a = Node('Paper', name=str(paper_list[i][0]), **feature)
    graph.create(a)
```

但通过测试发现，这种方法在进行大量节点的添加时，速度很慢（本文添加2708个带有1435个属性的节点使用了近两分钟）。在查阅官方文档和一些资料后，发现可以使用事务的方式，统一创建子图，然后再添加到图数据库中（待完成）

#### 添加关系

从`cora.cite`中得到论文引用关系`cite`之后，通过查询指定名称的节点，然后为其创建相应的引用关系，最后调用create方法添加关系

```python
for cite_list in cites:
    paper1, paper2 = cite_list[0], cite_list[1]
    a = matcher.match("Paper").where(f"_.name='{paper1}'").first()
    b = matcher.match("Paper").where(f"_.name='{paper2}'").first()
    rel = Relationship(b, "CITE", a)
    graph.create(rel)
```

测试发现，因为需要先查询两次节点的原因，关系的添加很慢，15分钟只添加了2000条左右的关系，需要优化。

#### 存储优化

在`cora.cite`中，是按照被引文献进行排序的，即引用同一篇文献的引用关系放在一起，例如文件的前十行均为paperID为35的文章

```shell
['35', '1033']
['35', '103482']
['35', '103515']
['35', '1050679']
['35', '1103960']
['35', '1103985']
['35', '1109199']
['35', '1112911']
['35', '1113438']
['35', '1113831']
```

因此可以在创建节点时，直接创建关系，然后一起添加到图数据库中。

整体思路是：按照引用关系`[paper1, paper2]`进行遍历，首先判断paper1是否改变，若不变则省去一次创建节点的次数；然后判断paper2是否已经在数据库中，若不在则创建，若在则查询；最后根据两个节点建立相应的关系。

优化效果很好，存储用时从原来的近1小时降低到10分钟左右。

后续可以使用事务和子图机制进一步缩短存储时间。

```python
import numpy as np
from py2neo import Graph, Node, Relationship, NodeMatcher

graph = Graph("http://localhost//:7474", username="neo4j", password="123456")
matcher = NodeMatcher(graph)

content_path = './cora/cora.content'
cite_path = './cora/cora.cites'

with open(content_path, "r") as f:
    contents = f.readlines()
with open(cite_path, "r") as f:
    cites = f.readlines()

contents = np.array([line.strip().split('\t') for line in contents])
paper_list, feature_list, label_list = np.split(contents, [1, -1], axis=1)
feature_list = np.concatenate((feature_list, label_list), axis=1)
paper_list = np.squeeze(paper_list)
cites = [line.strip().split("\t") for line in cites]  # 5429

node_dict = dict([(key, 0) for val, key in enumerate(paper_list)])  # 校验paper是否已添加入数据库
paper_dict = dict([(key, val) for val, key in enumerate(paper_list)])  # 生成PaperID到矩阵行数的映射

citedID = '0'
a_node = Node()
b_node = Node()
graph.delete_all()  # 清空图数据库
index = 0

for cite_list in cites:
    paper1, paper2 = cite_list[0], cite_list[1]
    # 先后创建或查询a_node和b_node
    if paper1 != citedID:  # a_node发生改变，需要查找或创建
        if node_dict[paper1] == 0:  #a_node不在数据库中，需要创建
            feature = dict([(str(val), key) for val, key in enumerate(feature_list[paper_dict[paper1]])])
            a_node = Node("Paper", name=paper1, **feature)
            graph.create(a_node)
            node_dict[paper1] = 1
        else:  # a_node在数据库中，需要查找
            a_node = matcher.match("Paper").where(f"_.name='{paper1}'").first()
        citedID = paper1
    # a_node已有，只需查找或创建b_node
    if node_dict[paper2] == 0:  # b_node不在数据库中，需要创建
        feature = dict([(str(val), key) for val, key in enumerate(feature_list[paper_dict[paper2]])])
        b_node = Node("Paper", name=paper2, **feature)
        graph.create(b_node)
        node_dict[paper2] = 1
    else:  # b_node在数据库中，需要查找
        b_node = matcher.match("Paper").where(f"_.name='{paper2}'").first()
    # 得到关系的两个对应节点后，创建关系
    rel = Relationship(b_node, "CITE", a_node)  # 创建关系
    graph.create(rel)

    if index%100 == 0:
        print(f"{index} relations have been added")
    index += 1
```

## GCN对边的学习效果

Link Prediction：《Modeling Relational Data with Graph Convolutional Networks》

DataSet：FB15k-237

Model：R-GCN(encoder) + DistMult(decoder)

Loss：cross-entropy

复现代码仓库：https://github.com/MichSchli/RelationPrediction

环境：python3.5 + tensorflow 1.4.1 (RTX 2080 Ti * 2)

### gcn_block.exp

```shell
(tf-gnn) user-lqz@admin:~/workspace/FangHonglin/cuda_test/RelationPrediction$ bash run-train.sh /settings/gcn_block.exp
WARNING (theano.configdefaults): install mkl with `conda install mkl-service`: No module named 'mkl'
{'Decoder': {'Name': 'bilinear-diag', 'RegularizationParameter': '0.01'}, 'Encoder': {'InternalEncoderDimension': '500', 'Concatenation': 'Yes', 'StoreEdgeData': 'No', 'SkipConnections': 'None', 'Name': 'gcn_basis', 'DiagonalCoefficients': 'No', 'PartiallyRandomInput': 'No', 'UseOutputTransform': 'No', 'AddDiagonal': 'No', 'UseInputTransform': 'Yes', 'RandomInput': 'No', 'DropoutKeepProbability': '0.8', 'NumberOfLayers': '2', 'NumberOfBasisFunctions': '100'}, 'Evaluation': {'Metric': 'MRR'}, 'Shared': {'CodeDimension': '500'}, 'Optimizer': {'Algorithm': {'Name': 'Adam', 'learning_rate': '0.01'}, 'ReportTrainLossEvery': '100', 'EarlyStopping': {'CheckEvery': '2000', 'BurninPhaseDuration': '6000'}, 'MaxGradientNorm': '1'}, 'General': {'ExperimentName': 'models/GcnBlock', 'GraphSplitSize': '0.5', 'NegativeSampleRate': '10', 'GraphBatchSize': '30000'}}
272115
[<tf.Tensor 'graph_edges:0' shape=(?, 3) dtype=int32>, <tf.Tensor 'Placeholder_1:0' shape=(?, 3) dtype=int32>, <tf.Tensor 'Placeholder:0' shape=(?,) dtype=float32>]
2020-07-31 21:01:18.316900: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2020-07-31 21:01:18.558220: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties:
name: GeForce RTX 2080 Ti major: 7 minor: 5 memoryClockRate(GHz): 1.545
pciBusID: 0000:82:00.0
totalMemory: 10.76GiB freeMemory: 10.45GiB
2020-07-31 21:01:18.753845: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 1 with properties:
name: GeForce RTX 2080 Ti major: 7 minor: 5 memoryClockRate(GHz): 1.545
pciBusID: 0000:83:00.0
totalMemory: 10.76GiB freeMemory: 10.45GiB
2020-07-31 21:01:18.754027: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Device peer to peer matrix
2020-07-31 21:01:18.754105: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1051] DMA: 0 1
2020-07-31 21:01:18.754115: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1061] 0:   Y N
2020-07-31 21:01:18.754122: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1061] 1:   N Y
2020-07-31 21:01:18.754135: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0)-> (device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:82:00.0, compute capability: 7.5)
2020-07-31 21:01:18.754143: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:1)-> (device: 1, name: GeForce RTX 2080 Ti, pci bus id: 0000:83:00.0, compute capability: 7.5)
SampleTransformer
GradientClipping
Adam
TrainLossReporter
EarlyStopper
ModelSaver
WARNING:tensorflow:From /home/user-lqz/anaconda3/envs/tf-gnn/lib/python3.5/site-packages/tensorflow/python/util/tf_should_use.py:107: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
Instructions for updating:
Use `tf.global_variables_initializer` instead.
Initial loss: 1.25448
Average train loss for iteration 1-100: 0.267123520225
Average train loss for iteration 101-200: 0.148406568617
Average train loss for iteration 201-300: 0.122154484317
```

最后依然OOM爆显存了.....

根据Github [Issue#8](https://github.com/MichSchli/RelationPrediction/issues/8) **[@rayrayraykk](https://github.com/rayrayraykk)**的方法，将`gcn_block.exp`中的`GraphBatchSize`从30000改为300，`InternalEncoderDimension`和`CodeDimension`从500改为200，模型可以运行，虽然还是调用的CPU计算，但计算速度大幅增快。

```shell
WARNING (theano.configdefaults): install mkl with `conda install mkl-service`: No module named 'mkl'
{'Decoder': {'Name': 'bilinear-diag', 'RegularizationParameter': '0.01'}, 'Encoder': {'UseOutputTransform': 'No', 'DropoutKeepProbability': '0.8', 'StoreEdgeData': 'No', 'NumberOfBasisFunctions': '100', 'Name': 'gcn_basis', 'UseInputTransform': 'Yes', 'InternalEncoderDimension': '200', 'AddDiagonal': 'No', 'DiagonalCoefficients': 'No', 'Concatenation': 'Yes', 'PartiallyRandomInput': 'No', 'SkipConnections': 'None', 'NumberOfLayers': '2', 'RandomInput': 'No'}, 'General': {'GraphBatchSize': '300', 'ExperimentName': 'models/GcnBlock', 'NegativeSampleRate': '10', 'GraphSplitSize': '0.5'}, 'Evaluation': {'Metric': 'MRR'}, 'Shared': {'CodeDimension': '200'}, 'Optimizer': {'ReportTrainLossEvery': '100', 'EarlyStopping': {'BurninPhaseDuration': '6000', 'CheckEvery': '2000'}, 'MaxGradientNorm': '1', 'Algorithm': {'Name': 'Adam', 'learning_rate': '0.01'}}}
272115
[<tf.Tensor 'graph_edges:0' shape=(?, 3) dtype=int32>, <tf.Tensor 'Placeholder_1:0' shape=(?, 3) dtype=int32>, <tf.Tensor 'Placeholder:0' shape=(?,) dtype=float32>]
2020-08-01 18:32:23.568991: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2020-08-01 18:32:23.943096: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
name: GeForce RTX 2080 Ti major: 7 minor: 5 memoryClockRate(GHz): 1.545
pciBusID: 0000:82:00.0
totalMemory: 10.76GiB freeMemory: 10.60GiB
2020-08-01 18:32:24.130528: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 1 with properties: 
name: GeForce RTX 2080 Ti major: 7 minor: 5 memoryClockRate(GHz): 1.545
pciBusID: 0000:83:00.0
totalMemory: 10.76GiB freeMemory: 10.60GiB
2020-08-01 18:32:24.130716: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Device peer to peer matrix
2020-08-01 18:32:24.130816: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1051] DMA: 0 1 
2020-08-01 18:32:24.130834: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1061] 0:   Y N 
2020-08-01 18:32:24.130844: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1061] 1:   N Y 
2020-08-01 18:32:24.130860: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:82:00.0, compute capability: 7.5)
2020-08-01 18:32:24.130876: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:1) -> (device: 1, name: GeForce RTX 2080 Ti, pci bus id: 0000:83:00.0, compute capability: 7.5)
SampleTransformer
GradientClipping
Adam
TrainLossReporter
EarlyStopper
ModelSaver
WARNING:tensorflow:From /home/user-lqz/anaconda3/envs/tf-gnn/lib/python3.5/site-packages/tensorflow/python/util/tf_should_use.py:107: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
Instructions for updating:
Use `tf.global_variables_initializer` instead.
Initial loss: 0.733639
Average train loss for iteration 1-100: 0.398559064865
Average train loss for iteration 101-200: 0.314240244925
Average train loss for iteration 201-300: 0.283124372214
Average train loss for iteration 301-400: 0.262287636399
Average train loss for iteration 401-500: 0.244079045802
Average train loss for iteration 501-600: 0.224757089317
Average train loss for iteration 601-700: 0.217351132631
Average train loss for iteration 701-800: 0.20891327545
Average train loss for iteration 801-900: 0.197149130404
Average train loss for iteration 901-1000: 0.186215469241
Average train loss for iteration 1001-1100: 0.179507168084
Average train loss for iteration 1101-1200: 0.181720136255
Average train loss for iteration 1201-1300: 0.180589640588
Average train loss for iteration 1301-1400: 0.174662291855
Average train loss for iteration 1401-1500: 0.166378410906
Average train loss for iteration 1501-1600: 0.161896246821
Average train loss for iteration 1601-1700: 0.161435388848
Average train loss for iteration 1701-1800: 0.159836556613
Average train loss for iteration 1801-1900: 0.162233484685
	    Raw	    Filtered
MRR	    0.012	0.012
H@1	    0.003	0.003
H@3	    0.008	0.008
H@10	0.021	0.022
Tested validation score at iteration 2000. Result: 0.0115082829409
saving...
Average train loss for iteration 1901-2000: 0.151952287182
Average train loss for iteration 2001-2100: 0.152650370747
Average train loss for iteration 2101-2200: 0.149650777802
Average train loss for iteration 2201-2300: 0.149231404066
Average train loss for iteration 2301-2400: 0.150278929994
Average train loss for iteration 2401-2500: 0.151916629821
Average train loss for iteration 2501-2600: 0.143547384217
Average train loss for iteration 2601-2700: 0.143923636898
Average train loss for iteration 2701-2800: 0.141646012366
Average train loss for iteration 2801-2900: 0.141806647405
Average train loss for iteration 2901-3000: 0.142812938094
Average train loss for iteration 3001-3100: 0.140599277988
Average train loss for iteration 3101-3200: 0.13751272589
Average train loss for iteration 3201-3300: 0.139074690044
Average train loss for iteration 3301-3400: 0.137135729194
Average train loss for iteration 3401-3500: 0.136827448159
Average train loss for iteration 3501-3600: 0.1353039518
Average train loss for iteration 3601-3700: 0.13576267533
Average train loss for iteration 3701-3800: 0.138574075177
Average train loss for iteration 3801-3900: 0.135007323623
	    Raw	    Filtered
MRR	    0.017	0.018
H@1	    0.006	0.006
H@3	    0.013	0.014
H@10	0.031	0.033
Tested validation score at iteration 4000. Result: 0.0173423935426
saving...
Average train loss for iteration 3901-4000: 0.135105981305
Average train loss for iteration 4001-4100: 0.136301774904
Average train loss for iteration 4101-4200: 0.133662451655
Average train loss for iteration 4201-4300: 0.135636581704
Average train loss for iteration 4301-4400: 0.132769066095
Average train loss for iteration 4401-4500: 0.133558651209
Average train loss for iteration 4501-4600: 0.133253392652
Average train loss for iteration 4601-4700: 0.134220164716
Average train loss for iteration 4701-4800: 0.132240785509
Average train loss for iteration 4801-4900: 0.132114236206
Average train loss for iteration 4901-5000: 0.130822513774
Average train loss for iteration 5001-5100: 0.130252244174
Average train loss for iteration 5101-5200: 0.130456224829
Average train loss for iteration 5201-5300: 0.132644971162
Average train loss for iteration 5301-5400: 0.128310849592
Average train loss for iteration 5401-5500: 0.129062893018
Average train loss for iteration 5501-5600: 0.131218217239
Average train loss for iteration 5601-5700: 0.128327239677
Average train loss for iteration 5701-5800: 0.128901169524
Average train loss for iteration 5801-5900: 0.127758514881
		Raw		Filtered
MRR		0.02	0.02
H@1		0.007	0.007
H@3		0.016	0.016
H@10	0.038	0.04
Tested validation score at iteration 6000. Result: 0.019700886744
saving...
Average train loss for iteration 5901-6000: 0.128414031491
Average train loss for iteration 6001-6100: 0.128678858802
Average train loss for iteration 6101-6200: 0.129787306339
Average train loss for iteration 6201-6300: 0.130661423728
Average train loss for iteration 6301-6400: 0.125296980664
Average train loss for iteration 6401-6500: 0.126010311395
Average train loss for iteration 6501-6600: 0.127962349132
Average train loss for iteration 6601-6700: 0.127901101038
Average train loss for iteration 6701-6800: 0.125632699803
Average train loss for iteration 6801-6900: 0.125565705001
Average train loss for iteration 6901-7000: 0.126050799638
Average train loss for iteration 7001-7100: 0.123286731467
Average train loss for iteration 7101-7200: 0.129371566772
Average train loss for iteration 7201-7300: 0.13400388822
Average train loss for iteration 7301-7400: 0.127659037486
Average train loss for iteration 7401-7500: 0.125996452421
Average train loss for iteration 7501-7600: 0.12586839363
Average train loss for iteration 7601-7700: 0.124882008061
Average train loss for iteration 7701-7800: 0.124159323312
Average train loss for iteration 7801-7900: 0.124195801616
		Raw		Filtered
MRR		0.022	0.023
H@1		0.008	0.008
H@3		0.019	0.019
H@10	0.042	0.044
Tested validation score at iteration 8000. Result: 0.022275383556
saving...
Average train loss for iteration 7901-8000: 0.127038534582
Average train loss for iteration 8001-8100: 0.124251663908
Average train loss for iteration 8101-8200: 0.124876790196
Average train loss for iteration 8201-8300: 0.12558349967
Average train loss for iteration 8301-8400: 0.124531253949
Average train loss for iteration 8401-8500: 0.125098097101
Average train loss for iteration 8501-8600: 0.124462078065
Average train loss for iteration 8601-8700: 0.1262208803
Average train loss for iteration 8701-8800: 0.126135697663
Average train loss for iteration 8801-8900: 0.125756008253
Average train loss for iteration 8901-9000: 0.125989745408
Average train loss for iteration 9001-9100: 0.125483163223
Average train loss for iteration 9101-9200: 0.124200710543
Average train loss for iteration 9201-9300: 0.122691995502
Average train loss for iteration 9301-9400: 0.122893726453
Average train loss for iteration 9401-9500: 0.125107270852
Average train loss for iteration 9501-9600: 0.126149047986
Average train loss for iteration 9601-9700: 0.124828136191
Average train loss for iteration 9701-9800: 0.123932513446
Average train loss for iteration 9801-9900: 0.123825722784
		Raw		Filtered
MRR		0.029	0.032
H@1		0.012	0.012
H@3		0.025	0.028
H@10	0.055	0.059
Tested validation score at iteration 10000. Result: 0.032109358205
saving...
Average train loss for iteration 9901-10000: 0.122377600595
Average train loss for iteration 10001-10100: 0.122479913607
Average train loss for iteration 10101-10200: 0.126490946785
Average train loss for iteration 10201-10300: 0.131784814745
Average train loss for iteration 10301-10400: 0.124623856246
Average train loss for iteration 10401-10500: 0.126125487313
Average train loss for iteration 10501-10600: 0.124828394502
Average train loss for iteration 10601-10700: 0.124754978046
Average train loss for iteration 10701-10800: 0.126575767472
Average train loss for iteration 10801-10900: 0.124978147447
Average train loss for iteration 10901-11000: 0.124048930407
Average train loss for iteration 11001-11100: 0.125314957276
Average train loss for iteration 11101-11200: 0.12510120675
Average train loss for iteration 11201-11300: 0.123760017082
Average train loss for iteration 11301-11400: 0.129462359548
Average train loss for iteration 11401-11500: 0.125138808787
Average train loss for iteration 11501-11600: 0.121126376614
Average train loss for iteration 11601-11700: 0.125720160753
Average train loss for iteration 11701-11800: 0.124498691037
Average train loss for iteration 11801-11900: 0.124156257957
		Raw		Filtered
MRR		0.027	0.029
H@1		0.009	0.01
H@3		0.025	0.027
H@10	0.056	0.059
Tested validation score at iteration 12000. Result: 0.0279832806244
Stopping criterion reached.
Stopping training.
```

### FB15k-237数据集简介

FB15k-237是FreeBase数据集的一个子集，包含237种关系和14k种实体。

|   类别   |  数量   |
| :------: | :-----: |
| Relation |   237   |
|  Entity  | 14,541  |
|  Train   | 271,115 |
|  Valid   | 17,535  |
|   Test   | 20,466  |

FreeBase是一个采用结构化数据的大型合作知识库，2014年被Google关闭，但由于其数据整体设计完善，常用来作为知识图谱方面研究的评价数据集。