import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GCNConv

import random
import numpy as np
import pickle

from loader import *
from utils import *


class PruneGAT(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5):
        super(PruneGAT, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        
        self.W = nn.Parameter(torch.full(size=(in_dim, out_dim), fill_value=0.5))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # self.a = nn.Parameter(torch.zeros(size=(2*out_dim, 1)))
        self.a = nn.Parameter(torch.full(size=(2*out_dim, 1), fill_value=0.2))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, data, edges_prob, classnames, device):
        a = self.a.clone()
        X, A = data.x, data.edge_index
        # print(f"X : {X.shape}, A: {A.shape}")

        h_i = torch.matmul(X, self.W)
        # h_i_prime = torch.zeros(len(classnames), h_i.shape[1])
        h_i_prime = []

        for classname in classnames:
            i = data.y.index(classname) # class index
            neighbors = A[1][A[0] == i]  # Get indices of neighbors

            if neighbors.numel() <= 0: 
                h_j = torch.randn(1, h_i.shape[1]).squeeze()  # 随机初始化一个非零的h_j
                h_j = h_j.unsqueeze(0).to(device)
                # print(f"48: {h_j.shape}")
                h_i_prime.append(h_j)

            else:
                pruning_prob = torch.tensor([edges_prob[(i, neighbor.item())] for neighbor in neighbors])
 
                max_prob_index = torch.argmax(pruning_prob)
                pruned_neighbor = neighbors[max_prob_index]

                h_j = h_i[pruned_neighbor].unsqueeze(0) # (1, 512)
                alpha = torch.cat([h_i[i].expand(h_j.shape[0], -1), h_j], dim=1)  # (1, 1024)
                alpha = torch.matmul(alpha, a) # (1, 1)
                alpha = F.leaky_relu(alpha, negative_slope=0.2)
                alpha = F.softmax(alpha, dim=0)
                h_j_prime = torch.sum(alpha.unsqueeze(1) * h_j, dim=0)
                h_j_prime = h_j_prime.to(device)
                h_i_prime.append(h_j_prime)
            
        h_i_prime = torch.stack(h_i_prime).to(device)
        h_i_prime = F.dropout(h_i_prime, p=self.dropout, training=self.training)
        h_i_prime = F.elu(h_i_prime) # （batch size, out_dim）

        h_i_prime = h_i_prime.squeeze(1)

        return h_i_prime


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
    

class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_size):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return x


def unsupervised_loss(z, edge_index):
    adj_scores = torch.mm(z, z.t()) 
    adj_mask = 1 - torch.eye(adj_scores.shape[0])  

    min_value = adj_scores.min()
    max_value = adj_scores.max()
    adj_scores = (adj_scores - min_value) / (max_value - min_value)

    adj_scores = adj_scores * adj_mask  
    adj_pred = torch.sigmoid(adj_scores)  
    
    num_nodes = edge_index.max().item() + 1
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    adj_matrix[edge_index[0], edge_index[1]] = 1
    
    loss = F.binary_cross_entropy(adj_pred, adj_matrix) 
    return loss

def negative_sampling(edge_index, num_neg_samples):
    num_nodes = edge_index.max().item() + 1
    neg_samples = torch.randint(num_nodes, (2, edge_index.size(1) * num_neg_samples))
    return neg_samples


def graphsage_train(args, device, input_dim, train_data):
    model = GraphSAGE(input_dim, args.hidden_channels, args.out_channels, args.agg_fun)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader([train_data], batch_size=args.batch_size, shuffle=True)
    print(f"Number of train batches: {len(train_loader)}")

    for epoch in range(args.num_epochs):
        total_loss = 0

        for batch in train_loader:

            optimizer.zero_grad()
            
            x = model(batch)
            loss = unsupervised_loss(x, batch.edge_index)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()


    torch.save(model.state_dict(), args.root + args.cache + 'model_SAGE_' + args.data.split("/")[-2] + args.new_trip + '.pt')
    print("graph model saved.")
    

def graphsage_inference(args, input_dim, data, train_class, test_seen_classes, test_unseen_classes):
    print(f"infer seen classes {len(test_seen_classes)}, unseen classes: {len(test_unseen_classes)}")

    if args.g_type == "SAGE":
        model = GraphSAGE(input_dim, args.hidden_channels, args.out_channels, args.agg_fun)
    elif args.g_type == "GNN":
        model = GNN(num_features=data.x.size(1), hidden_size=input_dim)
    elif args.g_type == "GCN":
        model = None
    elif args.g_type == "GAT":
        model = None

    model.load_state_dict(
        torch.load(args.root + args.cache + 'model_' + args.g_type + '_' + args.data.split("/")[-2] + args.new_trip + '.pt'))
    model.eval()

    pre_embeds = {}
    out = model(data)
    for i in range(len(data.y)):
        pre_embeds[data.y[i]] = out[i]  

    test_seen_embs = {}
    for string in test_seen_classes:
        if string in pre_embeds:
            test_seen_embs[string] = pre_embeds[string]

    test_unseen_embs = {}
    for string in test_unseen_classes:
        if string in pre_embeds:
            test_unseen_embs[string] = pre_embeds[string]

    train_embs = {}
    for str in train_class:
        if str in pre_embeds:
            train_embs[str] = pre_embeds[str]

    return pre_embeds, train_embs, test_seen_embs, test_unseen_embs


def graph_infer(args, input_dim, data):
    if args.g_type == "SAGE":
        model = GraphSAGE(input_dim, args.hidden_channels, args.out_channels, args.agg_fun)
    elif args.g_type == "GNN":
        model = GNN(num_features=data.x.size(1), hidden_size=input_dim)
    elif args.g_type == "GCN":
        model = None
    elif args.g_type == "GAT":
        model = None

    model.load_state_dict(torch.load(args.root + args.cache + 'model_' + args.g_type + '_' + args.data.split("/")[-2] + '.pt'))
    model.eval()

    pre_embeds = {}
    out = model(data)
    for i in range(len(data.y)):
        pre_embeds[data.y[i]] = out[i]  

    return pre_embeds


def gnn_train(args, device, input_dim, train_data):
    model = GNN(num_features=train_data.x.size(1), hidden_size=input_dim)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.num_epochs):
        optimizer.zero_grad()
        output = model(train_data)
        loss = criterion(output, train_data.x)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), args.root + args.cache + 'model_GNN_' + args.data.split("/")[-2] + args.new_trip + '.pt')
    print("graph model saved.")




def graph_processor(args, train_class, test_seen_classes, test_unseen_classes, task):
    file = 'new_triplets.txt' if args.new_trip != "" else 'triplets.txt'
    file_path = args.root + args.data + file
    vertex_attrs, edge_attrs, edges, edge_attr = read_graph_data(file_path)

    model, tokenizer = attr_encoder(args.root + args.bert_name)
    data = init_ver_emd(vertex_attrs, edges, edge_attr, edge_attrs, model, tokenizer)
    input_dim = data.x.size(1)

    train_data = split_dataset(data, vertex_attrs, args.root + args.data + args.test_unseen_loc)
    label_neighbors, label_freq = gen_vertex_text(data, edges)
    
    if args.mode == "train":
        if args.g_type == "GNN": # "GNN, GNN, GAT, GCN, SAGE"
            gnn_train(args, args.device, input_dim, train_data)
        elif args.g_type == "GCN":
            model = None
        elif args.g_type == "GAT":
            model = None
        else:
            graphsage_train(args, args.device, input_dim, train_data)

    
    pre_embeds, train_embs, test_seen_embs, test_unseen_embs = graphsage_inference(args, 
                                            input_dim, data, train_class, test_seen_classes, test_unseen_classes)

    return pre_embeds, train_embs, test_seen_embs, test_unseen_embs, train_data, data, label_freq, label_neighbors


def kg_graph_train(args):
    kg_dir = args.root + args.data + args.kg
    dataset = args.data.strip().split("/")[-2]

    file_path = kg_dir + 'train_triplets.txt'
    vertex_attrs, edge_attrs, edges, edge_attr = read_graph_data(file_path)

    model, tokenizer = attr_encoder(args.root + args.bert_name)
    data = init_ver_emd(vertex_attrs, edges, edge_attr, edge_attrs, model, tokenizer)
    input_dim = data.x.size(1)

    if args.mode == "train":
        if args.g_type == "GNN": # "GNN, GNN, GAT, GCN, SAGE"
            gnn_train(args, args.device, input_dim, data)
        elif args.g_type == "GCN":
            model = None
        elif args.g_type == "GAT":
            model = None
        else:
            graphsage_train(args, args.device, input_dim, data)

    train_emb = graph_infer(args, input_dim, data)

    train_label_neighbors, train_label_freq = gen_vertex_text(data, edges)

    return train_emb, data, train_label_neighbors, train_label_freq


def kg_graph_infer(args):
    kg_dir = args.root + args.data + args.kg
    cache_dir = args.root + args.cache + "feat/"
    
    feat_file = cache_dir + "feat_" + args.g_type + "_test_" + args.data.split("/")[-2] + ".pt"
    if os.path.exists(feat_file):
        with open(feat_file, 'rb') as f:
            feat = pickle.load(f)
        test_emb = feat
        print("dataset features load finish.")

    else: 
        file_path = kg_dir + 'test_triplets.txt'
        vertex_attrs, edge_attrs, edges, edge_attr = read_graph_data(file_path)

        model, tokenizer = attr_encoder(args.root + args.bert_name)
        data = init_ver_emd(vertex_attrs, edges, edge_attr, edge_attrs, model, tokenizer)
        input_dim = data.x.size(1)
        
        test_emb = graph_infer(args, input_dim, data)

        test_label_neighbors, test_label_freq = gen_vertex_text(data, edges)

    return test_emb, data, test_label_neighbors, test_label_freq
        

