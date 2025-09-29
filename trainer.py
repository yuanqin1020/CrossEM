


import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

import random
import numpy as np

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, agg_fun):
        super(GraphSAGE, self).__init__()
        self.dropout = nn.Dropout(p = 0.1)
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=agg_fun)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=agg_fun)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        return x

def graphsage_inference(input_dim, data, test_classes):

    model = GraphSAGE(input_dim, 256, 128, "MEAN")
    model.load_state_dict(torch.load('/home/yuanqin/chenchi/model_SAGE_CUB_200.pt'))
    model.eval()

    pre_embeds = {}
    out = model(data)
    for i in range(len(data.y)):
        pre_embeds[data.y[i]] = out[i]  

    test_embs = {}
    for string in test_classes:
        if string in pre_embeds:
            test_embs[string] = pre_embeds[string]

    return pre_embeds, test_embs
