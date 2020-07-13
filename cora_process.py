import numpy as np

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

# cites数据整理
