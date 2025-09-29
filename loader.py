import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import SAGEConv
from transformers import BertTokenizer, BertModel
from torch_geometric.utils import train_test_split_edges
import numpy as np
import random



def attr_encoder(cache_dir):
    # Create a BERT tokenizer to convert vertex labels to indices
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir, local_files_only=True)
    # Create a BERT model to extract features from vertex labels
    model = BertModel.from_pretrained("bert-base-uncased", cache_dir=cache_dir, local_files_only=True,
                                      output_hidden_states=True)
    model.eval()
    print(f"BERT model load complete from {cache_dir}.")

    return model, tokenizer

def read_classes(data, file):
    task = data.strip().split("/")[-2]
    classes = []
    with open(file, 'r') as file:
        for line in file:
            line = line.strip()
            if task == "CUB_200":
                classes.append(line.split(".")[1].replace("_", " ").lower())
            elif task == "AWA2":
                classes.append(line.replace("+", " ").lower())
            elif task == "SUN":
                classes.append(line.replace("_", " ").lower())


    print(f"class [0]: {classes[0]}")
    
    return classes




# Read text file, Build edge list and node features
def read_graph_data(file_path):
    print(f"Read {file_path}")
    """
        vertex_attrs: dict{vertex label, vertex id}
        edge_attrs: dict{(start vertex id, end vertex id): edge id}
        edges: dict{(start vertex id, end vertex id): edge label}
        edge_attr: [edge id]
    """
    edges = {}
    edge_attr = []
    vertex_attrs = {}
    edge_attrs = {}
    vertex_id = 0
    edge_id = 0
    with open(file_path, 'r') as file:
        for line in file:
            start_vertex, edge, end_vertex = line.strip().lower().split('\t')
            start_vertex = start_vertex.replace("_", " ").strip()
            edge = edge.replace("_", " ").strip()
            end_vertex = end_vertex.replace("_", " ").strip()

            if start_vertex not in vertex_attrs:
                vertex_attrs[start_vertex] = vertex_id
                vertex_id += 1
            if end_vertex not in vertex_attrs:
                vertex_attrs[end_vertex] = vertex_id
                vertex_id += 1
            if (vertex_attrs[start_vertex], vertex_attrs[end_vertex]) not in edge_attrs:
                edge_attrs[(vertex_attrs[start_vertex], vertex_attrs[end_vertex])] = edge_id
                edges[(vertex_attrs[start_vertex], vertex_attrs[end_vertex])] = edge
                edge_id += 1

            # edges.append((vertex_attrs[start_vertex], vertex_attrs[end_vertex]))
            edge_attr.append(edge_attrs[(vertex_attrs[start_vertex], vertex_attrs[end_vertex])])


    return vertex_attrs, edge_attrs, edges, edge_attr


def get_ent_rel_mapping(path, scala):
    # 读取entity2text.txt文件
    entity2text = {}
    # /m/0145m        Afrobeat
    ent_file = 'entity2text.txt' if scala == "" else 'entity2text_1' + scala + ".txt"
    print(ent_file)
    with open(path + ent_file, 'r') as f:
        for line in f:
            entity_id, entity_name = line.strip().lower().split('\t')
            entity2text[entity_id] = entity_name
            if len(entity2text) == 1:
                print(entity2text)

    # 读取relation2text.txt文件
    relation2text = {}
    with open(path + 'relation2text.txt', 'r') as f:
        for line in f:
            relation_id, relation_name = line.strip().lower().split('\t')
            relation2text[relation_id] = relation_name

    return entity2text, relation2text


def handle_kg_data(path, entity2text, relation2text):
    files = ["train", "dev", "test"]
    for file in files:
        out = path + file + "_triplets.txt"
        if os.path.exists(out):
            continue
        else: 
            with open(path + file + '.tsv', 'r') as f:
                lines = f.readlines()

            # 生成triples.txt文件
            with open(out, 'w') as f:
                for line in lines:
                    entity1_id, relation_id, entity2_id = line.strip().lower().split('\t')
                    if entity1_id not in entity2text.keys() or entity2_id not in entity2text.keys():
                        continue
                    entity1_name = entity2text[entity1_id]
                    entity2_name = entity2text[entity2_id]
                    relation_name = relation2text[relation_id]
                    f.write(f"{entity1_name}\t{relation_name}\t{entity2_name}\n")
            print(f"write {file} triplets file into {out}")


def init_ver_emd(vertex_attrs, edges, edge_attr, edge_attrs, model, tokenizer):

    # Create vertex feature tensor using BERT predictions
    num_vertices = len(vertex_attrs)
    # Initialize a tensor to store vertex embeddings
    vertex_embeddings = torch.zeros(num_vertices, 768)
    print(f"The number of vertex is : {num_vertices}")

    for i, label in enumerate(vertex_attrs):
        # if i > 100: break 
        # print(label)

        # Add special tokens to vertex label
        marked_label = "[CLS] " + label + " [SEP]"
        # Tokenize vertex label
        tokenized_label = tokenizer.tokenize(marked_label)
        # Convert tokenized label to indices
        indexed_label = tokenizer.convert_tokens_to_ids(tokenized_label)
        # Convert indices to tensor
        label_tensor = torch.tensor([indexed_label])
        # print(f"label_tensor: {label_tensor}")
        # Extract features from vertex label using BERT model
        with torch.no_grad():
            outputs = model(label_tensor)
            hidden_states = outputs[2]

        # Get the last hidden state of the first token ([CLS]) as the vertex representation
        label_representation = hidden_states[-1][0][0]

        # Store vertex embedding in tensor
        vertex_embeddings[i] = label_representation
        # print(f"vertex {i} label {label}: {label_representation.shape}")

    vertex_embeddings = torch.tensor(vertex_embeddings).to(torch.float)
    vertex_features = vertex_embeddings / vertex_embeddings.norm(dim=-1, keepdim=True)
    # print(f"vertex_features shape: {vertex_features.shape}")

    # Create edge index and edge attribute tensors
    edge_index = torch.tensor(list(edges.keys()), dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    # print(f"edge_index shape: {edge_index.shape}")

    # Convert vertex attributes to a dictionary
    vertex_attrs_dict = {i: attr for attr, i in vertex_attrs.items()}

    # Convert edge attributes to a dictionary
    edge_attrs_dict = {i: attr for attr, i in edge_attrs.items()}

    # Create vertex attributes tensor
    vertex_attr = [vertex_attrs_dict[i] for i in range(len(vertex_attrs))]
    # print(f"vertex_attr: {vertex_attr}")
    # vertex_attr = torch.tensor(vertex_attr, dtype=torch.long)

    # Create PyTorch Geometric Data object
    data = Data(x=vertex_features, edge_index=edge_index, edge_attr=edge_attr, y=vertex_attr)

    print("data preprocessing is complete")

    return data



def bert_emd(sentences, model, tokenizer):

    num_embeds = len(sentences)
    # Initialize a tensor to store vertex embeddings
    embeddings = torch.zeros(num_embeds, 768)
    # print(f"the number of vertex is : {num_vertices}")

    for i, label in enumerate(sentences):

        # Add special tokens to vertex label
        marked_label = "[CLS] " + label + " [SEP]"
        # Tokenize label
        tokenized_label = tokenizer.tokenize(marked_label)
        # Convert tokenized label to indices
        indexed_label = tokenizer.convert_tokens_to_ids(tokenized_label)
        # Convert indices to tensor
        label_tensor = torch.tensor([indexed_label])
        # Extract features from vertex label using BERT model
        with torch.no_grad():
            outputs = model(label_tensor)
            hidden_states = outputs[2]

        # Get the last hidden state of the first token ([CLS]) as the vertex representation
        label_representation = hidden_states[-1][0][0]

        # Store vertex embedding in tensor
        embeddings[i] = label_representation

    embeddings = torch.tensor(embeddings).to(torch.float)
    features = embeddings / embeddings.norm(dim=-1, keepdim=True)

    return features


def bert_emd_avg(sentences, model, tokenizer):

    num_embeds = len(sentences)
    # Initialize a tensor to store vertex embeddings
    embeddings = torch.zeros(num_embeds, 768)
    # print(f"the number of vertex is : {num_vertices}")

    for i, label in enumerate(sentences):

        # Add special tokens to vertex label
        marked_label = "[CLS] " + label + " [SEP]"
        # Tokenize label
        tokenized_label = tokenizer.tokenize(marked_label)
        # Convert tokenized label to indices
        indexed_label = tokenizer.convert_tokens_to_ids(tokenized_label)
        # Convert indices to tensor
        label_tensor = torch.tensor([indexed_label])
        # Extract features from vertex label using BERT model
        with torch.no_grad():
            outputs = model(label_tensor)
            hidden_states = outputs[2]

        # Get the last hidden state of the first token ([CLS]) as the vertex representation
        label_representation = hidden_states[-1][0][0]

        # Store vertex embedding in tensor
        embeddings[i] = label_representation

    embeddings = torch.tensor(embeddings).to(torch.float)
    embeddings = torch.mean(embeddings, dim=0, keepdim=True)

    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    return embeddings


def split_dataset(data, vertex_attrs, test_unseen_classes):
    # test_class = []
    # with open(test_file, 'r') as file:
    #     for line in file:
    #         line = line.strip()
    #         test_class.append(line.split(".")[1].replace("_", " "))

    test_class = test_unseen_classes
    print(f"Number of test classes: {test_class}")

    train_data_mask = torch.ones(data.num_nodes, dtype=torch.bool)
    test_data_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    for i in range(len(data.y)):
        if data.y[i] in test_class:
            # print(f"{i}: {data.y[i]}")
            train_data_mask[i] = False
            test_data_mask[i] = True

    train_indices = torch.nonzero(train_data_mask, as_tuple=False).view(-1)
    # print(f"train_indices: {len(train_indices)}")

    train_data = Data(x=data.x[train_indices], y=[data.y[i] for i in train_indices])
    edge_index = torch.tensor([], dtype=torch.long)

    for i in range(data.edge_index.size(1)):
        src, tgt = data.edge_index[:, i]
        if src.item() in train_indices and tgt.item() in train_indices:
            edge_index = torch.cat([edge_index, data.edge_index[:, i].view(2, 1)], dim=1)

    train_data.edge_index = edge_index
    # 获取test_data中的唯一节点ID
    unique_nodes = torch.unique(edge_index)
    # print(f"unique_nodes:{len(unique_nodes)}")
    # 构建节点ID的映射字典
    node_mapping = {node.item(): new_id for new_id, node in enumerate(unique_nodes)}
    # print(f"node_mapping: {node_mapping}")

    # 重新映射data中的节点ID
    train_data.edge_index = torch.tensor([[node_mapping[id.item()] for id in row] for row in edge_index])
    # print(f"train_data.edge_index: {train_data.edge_index.shape}")

    # 重新映射节点特征中的节点ID
    indices = torch.tensor([value for key, value in node_mapping.items()])
    train_data.x = data.x[indices]
    labels = []
    for i in unique_nodes:
        for key, value in vertex_attrs.items():
            if value == i.item():
                labels.append(key)
    train_data.y = labels
    
    # print(f"train2 x: {train_data.x}")
    # print(f"train2 edge_index: {train_data.edge_index}")
    # print(f"train2 y: {train_data.y}")

    
    # test_indices = torch.nonzero(test_data_mask, as_tuple=False).view(-1)
    # # print(f"test_indices: {len(test_indices)}")

    # test_data = Data(x=data.x[test_indices], y=[data.y[i] for i in test_indices])
    # test_edge_index = torch.tensor([], dtype=torch.long)

    # for i in range(data.edge_index.size(1)):
    #     src, tgt = data.edge_index[:, i]
    #     if src.item() in test_indices and tgt.item() in test_indices:
    #         test_edge_index = torch.cat([test_edge_index, data.edge_index[:, i].view(2, 1)], dim=1)

    # test_data.edge_index = test_edge_index
    # # 获取test_data中的唯一节点ID
    # test_unique_nodes = torch.unique(test_edge_index)
    # # print(f"unique_nodes:{len(test_unique_nodes)}")
    # # 构建节点ID的映射字典
    # test_node_mapping = {node.item(): new_id for new_id, node in enumerate(test_unique_nodes)}
    # # print(f"node_mapping: {test_node_mapping}")

    # # 重新映射data中的节点ID
    # test_data.edge_index = torch.tensor([[test_node_mapping[id.item()] for id in row] for row in test_edge_index])
    # # print(f"train_data.edge_index: {train_data.edge_index.shape}")

    # # 重新映射节点特征中的节点ID
    # indices = torch.tensor([value for key, value in test_node_mapping.items()])
    # test_data.x = data.x[indices]
    # print(f"test_data.x :{test_data.x.shape}")
    # labels = []
    # for i in test_unique_nodes:
    #     for key, value in vertex_attrs.items():
    #         if value == i.item():
    #             labels.append(key)
    # test_data.y = labels

    return train_data

    
# 返回该文件夹下所有后缀为jpg的文件路径的列表
def get_jpg_files(folder_path):
    # 创建一个空列表，用于存储文件路径
    jpg_files = []
    ent = 0
    # 遍历文件夹下的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        ent += 1
        # if ent > 1000: break

        # 遍历每个文件
        for file in files:
            # 获取文件的绝对路径
            file_path = os.path.join(root, file)
            # 获取文件的后缀名
            _, extension = os.path.splitext(file_path)
            # 如果后缀名为.jpg，将文件路径添加到列表中
            if extension.lower() == ".jpg":
                jpg_files.append(file_path)

    # 返回文件路径列表
    return jpg_files

def read_image(path, train_classes, test_unseen_classes, test_seen_classes):
    train_img = []
    test_unseen_img = []
    test_seen_img = []
    print(f"path: {path}")
    task = path.strip().split("/")[-3]

    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            if task == "CUB_200":
                name = root.split("/")[-1].split(".")
                name = name[1] if len(name) == 2 else ""
                if name == "": continue
                name = name.replace("_", " ").lower()
            elif task == "AWA2":
                name = root.split("/")[-1]
                if "." in name: continue
                name = name.replace("+", " ").lower()
            elif task == "SUN":
                name = root.split("/")[-1]
                if "." in name: continue
                name = name.replace("_", " ").lower()

            if name in train_classes:
                for file in files:
                    # print(f"file:{file}, root: {root}, append: {os.path.join(root, file)}")
                    train_img.append(os.path.join(root, file))

            if name in test_unseen_classes:
                for file in files:
                    test_unseen_img.append(os.path.join(root, file))

            if name in test_seen_classes:
                for file in files:
                    test_seen_img.append(os.path.join(root, file))

    return train_img, test_unseen_img, test_seen_img

def read_image_path(path, dataset, entity2text, text):
    imgs = []
    print(f"path: {path}")

    if dataset in ["CUB_200", "AWA2", "SUN"]:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    # print(f"file:{file}, root: {root}, append: {os.path.join(root, file)}")
                    imgs.append(os.path.join(root, file))
    else:
        ent_list = extract_entities(entity2text, text, dataset)
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for directory in dirs:
                    if directory in ent_list:
                        dir_path = os.path.join(root, directory)
                        file_list = os.listdir(dir_path)
                        imgs.extend([os.path.join(dir_path, file) for file in file_list])

        
    print(f"img[0]: {imgs[0]}")

    return imgs

def extract_entities(entity2text, text, dataset):
    txt_to_id = {v: k for k, v in entity2text.items()}

    ent_list = []
    for txt in text:
        id = txt_to_id[txt]
        if dataset == "FB15k": 
            id = id[1:].replace("/", ".")
        else:
            id = "n" + id
        ent_list.append(id)

    return ent_list



def split_img(image_paths, ratio, seed):
    random.seed(seed)
    random.shuffle(image_paths)
    split_index = int(len(image_paths) * ratio)

    # 切分为train和test
    train_paths = image_paths[:split_index]
    test_paths = image_paths[split_index:]
    
    return train_paths, test_paths

