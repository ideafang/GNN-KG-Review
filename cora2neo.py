import numpy as np
from py2neo import Graph, Node, Relationship, NodeMatcher

graph = Graph("http://localhost//:7474", username="neo4j", password="123456")
matcher = NodeMatcher(graph)

# 添加节点
# print("adding node...")
# content_path = './cora/cora.content'
# with open(content_path, "r") as f:
#     contents = f.readlines()
#
# contents = np.array([line.strip().split('\t') for line in contents])
# paper_list, feature_list, label_list = np.split(contents, [1, -1], axis=1)
#
# graph.delete_all() # 清空图数据库
# '''
# 存储节点及属性到neo4j，每个节点有1435个属性
# 方法一：直接遍历创建节点（已实现，但速度慢）
# 用时统计：
# 100个nodes：      9.79s
# 1000个nodes：     47.24s
# 2708个nodes(all): 115.70s
# 方法二：使用事务批处理，生成子图再提交（ToDo）
# https://www.jianshu.com/p/febe8a248582
# '''
# for i in range(paper_list.shape[0]):
#     feature = dict([(str(val), key) for val, key in enumerate(feature_list[i])])
#     a = Node('Paper', name=str(paper_list[i][0]), **feature)
#     graph.create(a)
#
# print('success!')

# # 添加关系
# print("adding relationship...")
# matcher = NodeMatcher(graph)
# # node = matcher.match("Paper").where("_.name='1061127'").first()
# # print(node)
#
# cite_path = './cora/cora.cites'
# with open(cite_path, "r") as f:
#     cites = f.readlines()
#
# cites = [line.strip().split("\t") for line in cites] # 5429
# cites = cites[:10]
#
# '''
# 方法一：
# 遍历查询每个节点，然后构建关系，最后存储到neo4j
# 特别慢
# '''
#
# for cite_list in cites:
#     print(cite_list)
#     paper1, paper2 = cite_list[0], cite_list[1]
#     a = matcher.match("Paper").where(f"_.name='{paper1}'").first()
#     b = matcher.match("Paper").where(f"_.name='{paper2}'").first()
#     rel = Relationship(b, "CITE", a)
#     graph.create(rel)
#
# print("success!")

# 存储优化
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
