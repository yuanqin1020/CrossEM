
from PIL import Image
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data

import clip
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import math
import random
import os

from loader import attr_encoder, bert_emd


def resnet_image(image_path, transform, model, fc):

    image = Image.open(image_path)
    image = image.convert("RGB")
    image = transform(image)

    image = image.unsqueeze(0)
    
    with torch.no_grad():
        features = model(image)

    features = fc(features)
    features = features / features.norm(dim=-1, keepdim=True)
 
    return features


def image_patches(image_path, transform, model, fc):

    image = Image.open(image_path)
    image = image.convert("RGB")

    patch_size = 64  
    patches = []
    for i in range(0, image.size[0], patch_size):
        for j in range(0, image.size[1], patch_size):
            patch = image.crop((i, j, i+patch_size, j+patch_size))  
            
            patch = transform(patch)  
            patches.append(patch)

    patches = torch.stack(patches)

    with torch.no_grad():
        features = model(patches)

    features = fc(features)
    features = features / features.norm(dim=-1, keepdim=True)
 
    return features


def get_image_patch_embeddings(image_path, model, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    patch_size = 32
    patches = []
    embeddings = []
    window_size = 224
    stride = 112
    for top in range(0, image.shape[2], stride):
        for left in range(0, image.shape[3], stride):
            patch = image[:, :, top:top+window_size, left:left+window_size]
            patches.append(patch)
            with torch.no_grad():
                embedding = model.encode_image(patch)
                embeddings.append(embedding)


    embeddings = torch.cat(embeddings, dim=0)
    embeddings = F.normalize(embeddings, dim=1)

    return embeddings 


def get_model_path(args):
    path = f'model_{args.method}_{args.ctx_init}_{args.ctx_bais}_freq_{args.data.split("/")[-2]}.pt'

    return args.root + args.cache + path

def remove_leaf(emb_dict, data):
    print(f"Before romoving vertex number: {data.x.size(0)}")
    y, edge_index = data.y, data.edge_index
    
    out_degrees = torch.zeros(len(emb_dict))
    out_degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))

    zero_outdegree_indices = (out_degrees == 0).nonzero(as_tuple=False).view(-1)
    
    zero_outdegree_nodes = [data.y[i] for i in zero_outdegree_indices.tolist()]
    print(f"zero_outdegree_nodes: {len(zero_outdegree_nodes)}, zero[0]: {zero_outdegree_nodes[0]}")
    
    non_zero_indices = out_degrees.nonzero(as_tuple=False).view(-1)
    
    train_emb = {k: v for i, (k, v) in enumerate(emb_dict.items()) if i in non_zero_indices.tolist()}
    
    data.y = [y[i] for i in non_zero_indices.tolist()]
    data.x = data.x[non_zero_indices.tolist()]
    
    print(f"After romoving vertex number: {data.x.size(0)}")

    return list(train_emb.keys())

def get_trans_img(images, transform, visual):
    trans = {}
    transformed = []
    for im in images:
        image = Image.open(im)
        image = transform(image)
        trans[im] = visual(image)
        transformed.append(trans)

    return transformed


def get_name_clusters(ver_embeds, num_clusters):
    ver_names, ver_vectors = list(ver_embeds.keys()), list(ver_embeds.values())

    ver_reps = torch.stack(ver_vectors)
    ver_arrays = ver_reps.detach().numpy()

    cluster_alg = KMeans(n_clusters=num_clusters, n_init=10)
    clusters = cluster_alg.fit_predict(ver_arrays)

    cluster_names = {i: [] for i in range(num_clusters)}
    
    for i, cluster in enumerate(clusters):
        cluster_names[cluster].append(ver_names[i])

    cluster_list = []
    merge = {}
    for cluster, names in cluster_names.items():
        print("Cluster", {cluster}, ":", len(names))
        dict = {}
        for name in names:
            if len(names) < 10:
                merge[name] = ver_embeds[name]
            else:
                dict[name] = ver_embeds[name]
        
        if len(dict) != 0:
            cluster_list.append(dict)
    
    if len(merge) < 10:
        cluster_list[0].update(merge)
    else:
        cluster_list.append(merge)
    
    print(f"cluster_list size: {len(cluster_list)}")

    return cluster_names, cluster_list


def split_graph(ver_emb, num):
    print(f"Number of block: {num}")
    data_set = []
    
    keys = list(ver_emb.keys())
    length = len(keys)
    split_size = (length + num - 1) // num

    for i in range(0, length, split_size):
        split_keys = keys[i:i+split_size]
        split_dict = {key: ver_emb[key] for key in split_keys}
        data_set.append(split_dict)

    return data_set

def min_cut_partition(similarity_matrix, num_partitions):

    edge_weights = 1 - similarity_matrix

    adjacency_matrix = torch.zeros_like(edge_weights)
    adjacency_matrix[edge_weights > 0] = 1

    partitions = []
    for _ in range(num_partitions - 1):
        cut = torch.zeros(similarity_matrix.size(0))
        cut[0] = 1
        for _ in range(similarity_matrix.size(0) - 1):
            cut = torch.matmul(adjacency_matrix, cut)

        partition = torch.nonzero(cut).squeeze()
        partitions.append(partition)

        adjacency_matrix[partition] = 0
        adjacency_matrix[:, partition] = 0

    last_partition = torch.nonzero(1 - torch.sum(adjacency_matrix, dim=0)).squeeze()
    partitions.append(last_partition)

    return partitions


def get_neighbors(data, names):
    classes_neighbors = {}
    indices = [data.y.index(name) if name in data.y else -1 for name in names]

    for i in indices:
        neighbors = data.edge_index[1][data.edge_index[0] == i]
        neigh_attrs = [data.y[j] for j in neighbors]
        classes_neighbors[data.y[i]] = neigh_attrs

        classes_neighbors[data.y[i]].append(data.y[i]) 

    return classes_neighbors


def attr_image_maxpatch(file, root, dataset):
    import ast

    check = [root + ".../Green_Jay_0027_65783.jpg", ...]

    attr_image_patch = {}
    c_num = 0
    if dataset in ["CUB_200", "AWA2", "SUN"]:
        with open(file, "r") as lines:
            for line in lines:
                for keyword in check:
                    if keyword in line:
                        c_num += 1
                        break

                try:
                    # item = line.strip()
                    item = ast.literal_eval(line.strip())
                    image = item['image']
                    attribute = item['attribute']
                    max_value = item['max']
                    if attribute not in attr_image_patch.keys():
                        attr_image_patch[attribute] = {}
                    attr_image_patch[attribute][image] = max_value
                except Exception as e:
                    print(f"Error occurs in line: {line.strip()}, {e}")
                    continue
        print(f"Number of pairs to check and ignore: {c_num}")

    else:
        with open(file, "r") as lines:
            for line in lines:
                item = line.strip().split(", ")
                image = root + item[0]
                max_value = item[-1]

                if len(item) != 3:
                    attribute = " ".join(item[1:-1])
                else:
                    attribute = item[1]
                
                if attribute not in attr_image_patch.keys():
                    attr_image_patch[attribute] = {}
                attr_image_patch[attribute][image] = max_value
    
    return attr_image_patch



cla = ['white', 'solid', 'black', 'about the same as head', 'long-wings', 'laysan albatross']
imgs = ["Laysan_Albatross_0001_545.jpg", ...]
    
def get_classes_image_simi(classes_neighbors, attr_image_patch):
    classes_image = {}
    
    for node, neighbors in classes_neighbors.items():

        node_img = {}
        for attribute, image_val in attr_image_patch.items():
            if attribute not in neighbors:
                continue

            # node_img.update(image_val)
            for img, val in image_val.items():

                # img_check = img.split("/")[-1]

                if img not in node_img.keys():
                    node_img[img] = float(0)
                node_img[img] = node_img[img] + float(val) #  max(node_img[img], float(val))

        classes_image[node] = node_img
    print(f"Number of classes-images neighbor similarity: {len(classes_image)}")

    return classes_image



def handle_triples(input, output):
    if os.path.exists(output):
        return

    with open(input, 'r') as f_in, open(output, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split('\t')

            second_col_parts = parts[1].split('_')
            new_second_col = second_col_parts[0]
            new_third_col = ' '.join(second_col_parts[1:]) + ' in ' + parts[2].replace("_", " ")

            f_out.write(f"{parts[0]}\t{new_second_col}\t{new_third_col}\n")


def preprocess_classes_image(data, classnames, file, root, dataset):
    print(f"Patch file: {file}")
    attr_image_patch = attr_image_maxpatch(file, root, dataset)
    
    classes_neighbors = get_neighbors(data, classnames)
    classes_images_simi = get_classes_image_simi(classes_neighbors, attr_image_patch)

    return classes_images_simi


def block_graph(ver_emb, num, blocker):
    print(f"Number of block: {num}")

    if blocker == "random":
        data_set = split_graph(ver_emb, num)

    elif blocker == "cluster":
        _, data_set = get_name_clusters(ver_emb, num)
    
    return data_set

def remove_unsimilar_images(simi_matrix, images, c_prune):
    to_remove = []

    norm = simi_matrix.norm(dim=-1, keepdim=True)
    norm[norm == 0] = 1  
    simi_norm = simi_matrix / norm
    
    for j in range(simi_norm.size(1)):
        column = simi_norm[:, j]
        # print(f"max and min in column {j}: {torch.max(column)}, {torch.min(column)}")
        for c in column:
            if c > c_prune: break # c > 5
            to_remove.append(j)
    new_simi_matrix = torch.index_select(
        simi_matrix, 1, torch.tensor([i for i in range(simi_matrix.size(1)) if i not in to_remove], dtype=torch.long))

    new_images = [images[i] for i in range(len(images)) if i not in to_remove]

    return new_simi_matrix, new_images


def remove_slight_images(simi_feat, theta_1, theta_2):

    variances = torch.var(simi_feat, dim=1)
    means = torch.mean(simi_feat, dim=1)
    filtered_simi_feat = simi_feat[(variances > theta_1) & (means > theta_2)]

    return filtered_simi_feat

def block_images(classes_images_simi, classnames, images, cluster_num, batch_size, c_prune, hns, theta_1, theta_2, bl):
    print(f"classnames, images size: {len(classnames)}, {len(images)}")

    simi_matrix = torch.empty(len(classnames), len(images))
    for i in range(len(classnames)):
        if classnames[i] not in classes_images_simi.keys():  continue

        for j in range(len(images)):
            if images[j] not in classes_images_simi[classnames[i]].keys():  continue
            simi_matrix[i, j] = classes_images_simi[classnames[i]][images[j]]

    if bl == 'pcp':
        images_list = pcp_block(images, simi_matrix, cluster_num, batch_size, c_prune, hns, theta_1, theta_2)
    elif bl == 'hash':
        images_list = hash_block(simi_matrix, images, c_prune)
    elif bl == "simi":
        images_list = similarity_block(simi_matrix, images, c_prune)

    return images_list


def similarity_block(simi_matrix, images, c_prune):
    images_list = []
    _, new_images = remove_unsimilar_images(simi_matrix, images, c_prune)
    # print(f"new_images: {new_images[:3]}, {len(new_images)}")
    images_list.append(new_images)

    return images_list


def hash_block(simi_matrix, images, c_prune):
    images_list = []
    new_simi_matrix, new_images = remove_unsimilar_images(simi_matrix, images, c_prune)
    simi_matrix = new_simi_matrix
    images = new_images
    print(f"After simi removing: classnames, images size: {new_simi_matrix.size(0)}, {new_simi_matrix.size(1)}")
    simi_matrix = torch.tensor(simi_matrix)

    # Perform LSH on the similarity matrix
    hashed_values, bucket_indices = LSH(simi_matrix, num_buckets=5)

    # Print the column indices for each bucket
    for bucket, indices in bucket_indices.items():
        print(f"Bucket {bucket}: {len(indices)}")
        imgs = [images[ind] for ind in indices]
        images_list.append(imgs)

    return images_list


def LSH(similarity_matrix, num_buckets):
    num_rows, num_cols = similarity_matrix.size()

    hashed_values = torch.zeros(num_rows, num_cols)

    threshold = 1.0 / num_buckets

    bucket_indices = {}

    for i in range(num_rows):
        for j in range(num_cols):
            hashed_value = int(similarity_matrix[i, j] // threshold)
            hashed_values[i, j] = hashed_value

            if hashed_value in bucket_indices:
                if j not in bucket_indices[hashed_value]:
                    bucket_indices[hashed_value].append(j)
            else:
                bucket_indices[hashed_value] = [j]

    return hashed_values, bucket_indices



def pcp_block(images, simi_matrix, cluster_num, batch_size, c_prune, hns, theta_1, theta_2):
    new_simi_matrix, new_images = remove_unsimilar_images(simi_matrix, images, c_prune)
    simi_matrix = new_simi_matrix
    images = new_images
    print(f"After simi removing: classnames, images size: {new_simi_matrix.size(0)}, {new_simi_matrix.size(1)}")

    if len(images) <= cluster_num: return []

    simi_matrix = torch.tensor(simi_matrix)

    simi_matrix[torch.isnan(simi_matrix)] = float('-100')
    simi_matrix[torch.isinf(simi_matrix)] = float('100')
    simi_feat = F.softmax(simi_matrix, dim=-1) 

    simi_feat = remove_slight_images(simi_feat, theta_1, theta_2)
    print(f"After slight removing: classnames, images size: {simi_feat.shape[0]}, {simi_feat.shape[1]}")

    n_clusters = cluster_num
    simi_feat_array = simi_feat.numpy().T  
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(simi_feat_array)

    cluster_labels = kmeans.labels_

    clusters = {}
    for i in range(len(cluster_labels)):
        label = cluster_labels[i]
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    adjusted_clusters = {}

    min_count = math.ceil(batch_size / 2)
    new_list = []
    for label, indices in clusters.items():
        print("Cluster", {label}, ":", len(indices))
        if len(indices) < min_count:
            new_list.extend(indices)
            if hns and len(new_list) > min_count:
                max_count = math.ceil(len(new_list) / batch_size) * batch_size
                add = random.sample(set(range(len(images))) - set(new_list), max_count - len(new_list))
                adjusted_clusters[len(clusters)] = new_list + add
                new_list = []
            else:
                adjusted_clusters[len(clusters)] = new_list
        else:
            max_count = math.ceil(len(indices) / batch_size) * batch_size

            if hns and max_count < len(images):
                additional_indices = random.sample(set(range(len(images))) - set(indices), max_count - len(indices))
                adjusted_clusters[label] = indices + additional_indices
            else:
                adjusted_clusters[label] = indices

    images_list = []
    for label, indices in adjusted_clusters.items():
        print("Adjusted Cluster", {label}, ":", len(indices))
        imgs = [images[ind] for ind in indices]
        images_list.append(imgs)
    
    return images_list


def batch_hard_negative(classnames, images, classes_images_simi, top_num):
    
    simi_matrix = torch.empty(len(classnames), len(images))

    for i in range(len(classnames)):
        if classnames[i] not in classes_images_simi.keys(): continue
        for j in range(len(images)):
            if images[j] not in classes_images_simi[classnames[i]].keys(): continue
            # if classnames[i] in cla:
            #     print(f"{classnames[i]}, {images[j]}, {classes_images_simi[classnames[i]][images[j]]}")

            simi_matrix[i, j] = classes_images_simi[classnames[i]][images[j]]

    topk_values, topk_indexes = torch.topk(simi_matrix.flatten(), k=top_num, largest=True)
    class_indexes = topk_indexes // simi_matrix.size(1)
    image_indexes = topk_indexes % simi_matrix.size(1)

    return class_indexes, image_indexes
    

def gen_vertex_text(data, edges):

    label_neighbors = {}
    label_freq = {}
    for ind in range(len(data.y)):
        vi = data.y[ind].replace("_", " ").lower()
        if vi in label_freq:
            label_freq[vi] += 1
        else:
            label_freq[vi] = 1

        text_list = []

        for edge, attribute in edges.items():
            i, j = edge
            if i == ind:
                vj_attr = data.y[j]
                edge_label = attribute

                label = vj_attr.lower().replace('_', ' ')
                text_list.append(label)

                if label in label_freq:
                    label_freq[label] += 1
                else:
                    label_freq[label] = 1

        label_neighbors[vi] = text_list

    print(f"label_neighbors size: {sum([len(v) for k,v in label_neighbors.items()])}")
    print(f"label freq size: {sum([v for k,v in label_freq.items()])}")

    return label_neighbors, label_freq


def gen_hard_prompt(label_neighbors):
    prompts = {}
    for key, values in label_neighbors.items():
        prompt = f"{key}"
        for str in values:
            prompt = f"{prompt} {str}"
        prompts[key] = prompt
    
    return prompts

def reconstruct_data(data, classnames, device):
    re_data = Data()
    data = data.to(device)
    temp_edge_index = torch.tensor([], dtype=torch.long).to(device)

    for i in range(data.edge_index.size(1)):
        src, tgt = data.edge_index[:, i]
        if data.y[src.item()] in classnames or data.y[tgt.item()] in classnames:
            temp_edge_index = torch.cat([temp_edge_index, data.edge_index[:, i].view(2, 1)], dim=1)

    print(f"temp_edge_index: {len(temp_edge_index[0])}")
    unique_nodes = torch.unique(temp_edge_index)

    node_mapping = {node.item(): new_id for new_id, node in enumerate(unique_nodes)}

    re_data.edge_index = torch.tensor([[node_mapping[id.item()] for id in row] for row in temp_edge_index])

    indices = torch.tensor([new for _, new in node_mapping.items()])
    re_data.x = data.x[indices]

    labels = []
    for old, new in node_mapping.items():
        labels.append(data.y[old])

    re_data.y = labels

    print(f"re_data.edge_index: {re_data.edge_index.shape}")
    print(f"re_data x: {re_data.x.shape}, re_data y: {len(re_data.y)}")

    return re_data


def edge_prune_probability(re_data, label_neighbors, label_freq):
    edge_freq = {}

    for ind in range(len(re_data.y)):
        classname = re_data.y[ind]
        neighbors = label_neighbors[classname]
        freqs = [label_freq[neigh] * 1.0 / sum(label_freq.values()) for neigh in neighbors]
        prune_probs = [1 - i / sum(freqs) for i in freqs]

        vi = ind
        for k in range(len(neighbors)):
            neigh = neighbors[k]
            if neigh in re_data.y:
                vj = re_data.y.index(neigh)
                edge_freq[(vi, vj)] = prune_probs[k]

    print(f"edge_freq size: {len(edge_freq)}")

    return edge_freq



