import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv

dataset = "./FB15k-237"
relations_path = dataset + '/relations.dict'
entities_path = dataset + '/entities.dict'
train_path = dataset + '/train.txt'
valid_path = dataset + '/valid.txt'
test_path = dataset + '/test.txt'

def read_triplets_as_list(filename, entity_dict, relation_dict):
    entity_dict = read_dictionary(entity_dict, id_lookup=False)
    relation_dict = read_dictionary(relation_dict, id_lookup=False)

    l = []
    for triplet in read_triplets(filename):
        entity_1 = entity_dict[triplet[0]]
        relation = relation_dict[triplet[1]]
        entity_2 = entity_dict[triplet[2]]

        l.append([entity_1, relation, entity_2])

    return l


def read_dictionary(filename, id_lookup=True):
    d = {}
    for line in open(filename, 'r+'):
        line = line.strip().split('\t')

        if id_lookup:
            d[int(line[0])] = line[1]
        else:
            d[line[1]] = int(line[0])

    return d

def read_triplets(filename):
    for line in open(filename, 'r+'):
        processed_line = line.strip().split('\t')
        yield processed_line


train_triplets = read_triplets_as_list(train_path, entities_path, relations_path)
valid_triplets = read_triplets_as_list(valid_path, entities_path, relations_path)
test_triplets = read_triplets_as_list(test_path, entities_path, relations_path)

train_data = np.array(train_triplets)
valid_data = np.array(valid_triplets)
test_data = np.array(test_triplets)

entities = read_dictionary(entities_path)
relations = read_dictionary(relations_path)

print(f"num_nodes: {len(entities)}")
print(f"num_rels: {len(relations)}")
print(f"train data shape {train_data.shape}")
print(f"valid data shape {valid_data.shape}")
print(f"test data shape {test_data.shape}")




