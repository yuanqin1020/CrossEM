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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir, local_files_only=True)
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


def read_graph_data(file_path):

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

            edge_attr.append(edge_attrs[(vertex_attrs[start_vertex], vertex_attrs[end_vertex])])


    return vertex_attrs, edge_attrs, edges, edge_attr


def get_ent_rel_mapping(path, scala):
    entity2text = {}
    ent_file = 'entity2text.txt' if scala == "" else 'entity2text_1' + scala + ".txt"
    print(ent_file)
    with open(path + ent_file, 'r') as f:
        for line in f:
            entity_id, entity_name = line.strip().lower().split('\t')
            entity2text[entity_id] = entity_name
            if len(entity2text) == 1:
                print(entity2text)

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

    num_vertices = len(vertex_attrs)
    vertex_embeddings = torch.zeros(num_vertices, 768)
    print(f"The number of vertex is : {num_vertices}")

    for i, label in enumerate(vertex_attrs):
        marked_label = "[CLS] " + label + " [SEP]"
        tokenized_label = tokenizer.tokenize(marked_label)
        indexed_label = tokenizer.convert_tokens_to_ids(tokenized_label)
        label_tensor = torch.tensor([indexed_label])
        with torch.no_grad():
            outputs = model(label_tensor)
            hidden_states = outputs[2]

        label_representation = hidden_states[-1][0][0]

        vertex_embeddings[i] = label_representation

    vertex_embeddings = torch.tensor(vertex_embeddings).to(torch.float)
    vertex_features = vertex_embeddings / vertex_embeddings.norm(dim=-1, keepdim=True)

    edge_index = torch.tensor(list(edges.keys()), dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    vertex_attrs_dict = {i: attr for attr, i in vertex_attrs.items()}
    edge_attrs_dict = {i: attr for attr, i in edge_attrs.items()}

    vertex_attr = [vertex_attrs_dict[i] for i in range(len(vertex_attrs))]
    data = Data(x=vertex_features, edge_index=edge_index, edge_attr=edge_attr, y=vertex_attr)

    print("data preprocessing is complete")

    return data



def bert_emd(sentences, model, tokenizer):

    num_embeds = len(sentences)
    embeddings = torch.zeros(num_embeds, 768)

    for i, label in enumerate(sentences):
        marked_label = "[CLS] " + label + " [SEP]"
        tokenized_label = tokenizer.tokenize(marked_label)
        indexed_label = tokenizer.convert_tokens_to_ids(tokenized_label)
        label_tensor = torch.tensor([indexed_label])
        with torch.no_grad():
            outputs = model(label_tensor)
            hidden_states = outputs[2]

        label_representation = hidden_states[-1][0][0]

        embeddings[i] = label_representation

    embeddings = torch.tensor(embeddings).to(torch.float)
    features = embeddings / embeddings.norm(dim=-1, keepdim=True)

    return features


def bert_emd_avg(sentences, model, tokenizer):

    num_embeds = len(sentences)
    embeddings = torch.zeros(num_embeds, 768)

    for i, label in enumerate(sentences):

        marked_label = "[CLS] " + label + " [SEP]"
        tokenized_label = tokenizer.tokenize(marked_label)
        indexed_label = tokenizer.convert_tokens_to_ids(tokenized_label)
        label_tensor = torch.tensor([indexed_label])
        with torch.no_grad():
            outputs = model(label_tensor)
            hidden_states = outputs[2]

        label_representation = hidden_states[-1][0][0]

        embeddings[i] = label_representation

    embeddings = torch.tensor(embeddings).to(torch.float)
    embeddings = torch.mean(embeddings, dim=0, keepdim=True)

    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    return embeddings


def split_dataset(data, vertex_attrs, test_unseen_classes):
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

    train_data = Data(x=data.x[train_indices], y=[data.y[i] for i in train_indices])
    edge_index = torch.tensor([], dtype=torch.long)

    for i in range(data.edge_index.size(1)):
        src, tgt = data.edge_index[:, i]
        if src.item() in train_indices and tgt.item() in train_indices:
            edge_index = torch.cat([edge_index, data.edge_index[:, i].view(2, 1)], dim=1)

    train_data.edge_index = edge_index
    unique_nodes = torch.unique(edge_index)
    node_mapping = {node.item(): new_id for new_id, node in enumerate(unique_nodes)}
    train_data.edge_index = torch.tensor([[node_mapping[id.item()] for id in row] for row in edge_index])
    indices = torch.tensor([value for key, value in node_mapping.items()])
    train_data.x = data.x[indices]
    labels = []
    for i in unique_nodes:
        for key, value in vertex_attrs.items():
            if value == i.item():
                labels.append(key)
    train_data.y = labels

    return train_data

    
def get_jpg_files(folder_path):
    jpg_files = []
    ent = 0
    for root, dirs, files in os.walk(folder_path):
        ent += 1

        for file in files:
            file_path = os.path.join(root, file)
            _, extension = os.path.splitext(file_path)
            if extension.lower() == ".jpg":
                jpg_files.append(file_path)

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

    train_paths = image_paths[:split_index]
    test_paths = image_paths[split_index:]
    
    return train_paths, test_paths

