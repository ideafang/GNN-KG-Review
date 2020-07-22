import numpy as np
from py2neo import Graph, Node, Relationship, NodeMatcher

graph = Graph("http://localhost//:7474", username="neo4j", password="123456")

# 添加节点
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

# 添加关系
matcher = NodeMatcher(graph)
# node = matcher.match("Paper").where("_.name='1061127'").first()
# print(node)

cite_path = './cora/cora.cites'
with open(cite_path, "r") as f:
    cites = f.readlines()

cites = [line.strip().split("\t") for line in cites]
# cites = cites[:10]
print(len(cites))  # 5429
'''
方法一：
遍历查询每个节点，然后构建关系，最后存储到neo4j
特别慢
'''

for cite_list in cites:
    paper1, paper2 = cite_list[0], cite_list[1]
    a = matcher.match("Paper").where(f"_.name='{paper1}'").first()
    b = matcher.match("Paper").where(f"_.name='{paper2}'").first()
    rel = Relationship(b, "CITE", a)
    graph.create(rel)


