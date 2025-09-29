
import numpy as np
import torch
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import *

def map_label(label, classes):
    mapped_label = torch.LongTensor(len(label))
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    
    return mapped_label


def map_img(img_path, data, ent2txt):
    task = data.strip().split("/")[-2]
    class_id = 0

    if task == "CUB_200":
        class_name = img_path.strip().split("/")[-2].split(".")[1].replace("_", " ").lower()
        class_id = int(img_path.strip().split("/")[-2].split(".")[0].strip()) - 1
    elif task == "AWA2":
        class_name = img_path.strip().split("/")[-2].replace("+", " ").lower()
    elif task == "SUN":
        items = img_path.strip().lower().split("/")
        if len(items[-3]) == 1:
            class_name = items[-2].replace("_", " ")
        else:
            class_name = items[-3] + " " + items[-2]
            class_name = class_name.replace("_", " ")
    elif task == "FB15k":
        # ent2txt: /m/0145m  Afrobeat
        ent = img_path.strip().split("/")[-2].replace("m.", "m/")
        class_name = ent2txt["/" + ent].lower()
    elif task == "WN18":
        ent = img_path.strip().split("/")[-2]
        class_name = ent2txt[ent[1:]].lower()
    
    return class_name, class_id

def true_targets(image_paths, data, ent2txt=None):
    true_class_img = {}

    for i in range(len(image_paths)):
        name, id = map_img(image_paths[i], data, ent2txt)
        
        if name not in true_class_img.keys():
            true_class_img[name] = []
        true_class_img[name].append(image_paths[i])

    double_check(true_class_img)

    return true_class_img

def double_check(true_class_img):
    with open("./data.txt", "w") as file:
        for name, paths in true_class_img.items():
            for path in paths:
                file.write(f"{name} \t {path}\n")




def sampling_constractive_loss(logits, batch_classes, batch_path, classes_images_simi, sample_num, margin = 0.2):
    class_indexes, image_indexes = batch_hard_negative(batch_classes, batch_path, classes_images_simi, sample_num)
    # print(f"Class Index: {class_indexes[i].item()}, Image Index: {image_indexes[i].item()}")

     # Get the top k indices with highest similarity for each text
    top_k_values, top_k_indices = torch.topk(logits, k=sample_num, dim=1)

    # Generate positive mask
    positive_mask = torch.zeros_like(logits)
    for i in range(len(top_k_indices)):
        # positive_mask.scatter_(1, top_k_indices[:, i].unsqueeze(1), 1)
        class_idx = class_indexes[i].item()
        image_idx = image_indexes[i].item()
        if class_idx < logits.shape[0] and image_idx < logits.shape[1]:
            positive_mask[class_idx, image_idx] = 1
    
    # Generate negative mask for pairs not in top k 
    # 值为1表示对应位置的样本是负样本，值为0表示对应位置的样本不是负样本
    negative_mask = torch.ones_like(logits)
    for i in range(len(top_k_indices)):
        class_idx = class_indexes[i].item()
        image_idx = image_indexes[i].item()
        if image_idx in top_k_indices[class_idx]:
            negative_mask[class_idx, image_idx] = 0
        # print(f"Negetive example: {batch_classes[class_idx]}, {batch_path[image_idx]}")
    

    # 将正样本和负样本分别与相似度矩阵相乘，得到正负样本对应的相似度
    positive_logits = torch.sum(logits * positive_mask, dim=1)
    negative_logits = torch.sum(logits * negative_mask, dim=1)

    contrastive_loss = torch.mean(torch.max(torch.zeros_like(positive_logits), margin - positive_logits + negative_logits))

    return contrastive_loss

def compute_loss(logits, tokenized_prompts, oloss): 
    if oloss:
        loss = enhance_constractive_loss(logits, tokenized_prompts)
    else:
        loss = constractive_loss(logits)
    return loss


def enhance_constractive_loss(logits, prompt_matrix, margin = 0.2, orthogonal_weight=0.3):
    # 计算每张图片对应的相似度最高的top3文本
    if logits.dim() == 1:
        logits = torch.unsqueeze(logits, dim=0)
    if logits.shape[1] < 10: k = 1
    else: k = 3
    top_k_values, top_k_indices = torch.topk(logits, k=k, dim=1)

    # 生成正样本对和负样本对的掩码
    positive_mask = torch.zeros_like(logits)
    positive_mask.scatter_(1, top_k_indices[:, 0].unsqueeze(1), 1)
    negative_mask = 1 - positive_mask

    # 将正样本和负样本分别与相似度矩阵相乘，得到正负样本对应的相似度
    positive_logits = torch.sum(logits * positive_mask, dim=1)
    negative_logits = torch.sum(logits * negative_mask, dim=1)

    # 计算对比损失
    contrastive_loss = torch.mean(torch.max(torch.zeros_like(positive_logits), margin - positive_logits + negative_logits))

    # print(f"prompt_matrix: {prompt_matrix}")
    # 计算正交约束损失
    orthogonal_loss = torch.mean(torch.sqrt(
        torch.abs(torch.matmul(
            prompt_matrix.float(), prompt_matrix.t().float()) - torch.eye(prompt_matrix.size(0))
            .to(prompt_matrix).to(torch.float))))

    # 综合对比损失和正交约束损失
    total_loss = contrastive_loss + orthogonal_weight * orthogonal_loss
    # print(f"total_loss: {total_loss}")

    return total_loss


def constractive_loss(logits, margin = 0.2):
    """
        encourage similarity between positive samples and dissimilarity between negative samples. 
        to minimize the distance between positive samples and maximize the distance between negative samples.
    """
    # 计算每张图片对应的相似度最高的top3文本
    top_k_values, top_k_indices = torch.topk(logits, k=3, dim=1)

    # 生成正样本对和负样本对的掩码
    positive_mask = torch.zeros_like(logits)
    positive_mask.scatter_(1, top_k_indices[:, 0].unsqueeze(1), 1)
    negative_mask = 1 - positive_mask

    # 将正样本和负样本分别与相似度矩阵相乘，得到正负样本对应的相似度
    positive_logits = torch.sum(logits * positive_mask, dim=1)
    negative_logits = torch.sum(logits * negative_mask, dim=1)

    contrastive_loss = torch.mean(torch.max(torch.zeros_like(positive_logits), margin - positive_logits + negative_logits))

    return contrastive_loss

def cross_entory_loss(logits):
    criterion = nn.CrossEntropyLoss()
    # indices = torch.topk(logits, k=3, dim=1)[1]
    # # print(f"indices: {indices}")
    # labels = torch.zeros(logits.size(0), logits.size(1))
    # labels.scatter_(1, indices, 1)
    # labels = labels.to(args.device)
    # loss = criterion(logits, labels.long())

    # # 计算损失
    targets = torch.zeros_like(logits)  # 将所有样本标记为负样本
    targets[:, :1] = 1  # 将相似度最高的1个样本标记为正样本
    loss = criterion(logits, targets)

    return loss


def supervised_loss(logits, batch_path, classes, device, task):

    labels = np.zeros((len(classes), len(batch_path)), dtype=int)
    true_label = true_targets(batch_path, task)
    print(f"true_label : {true_label}")

    for i in range(len(classes)):
        if classes[i] not in true_label.keys(): 
            continue

        # paths = true_label[classes[i]]
        for j in range(len(batch_path)):
            name, id = map_img(batch_path[j], task)
            if name == classes[i]: 
                labels[i, j] = 1
                print(f"supervised class and labels: {classes[i]} - {batch_path[j]}")

    labels = torch.from_numpy(labels).float().to(device)

    # 计算概率分布
    # probs = torch.sigmoid(logits)
    probs = torch.zeros(logits.size(1), logits.size(0)).to(device)
    values, indices = torch.topk(logits.T, k=10, dim=1)
    probs.scatter_(1, indices, 1)
    probs = probs.T

    print(f"logits: {logits.shape}, probs: {probs.shape}, labels: {labels.shape}")
    print(f"probs: {probs.T}, labels: {labels.T}")
    
    return F.cross_entropy(probs, labels)


"""
    CZSL: predict unseen; GZSL: predict seen and unseen
    计算每个类别的准确率和总体准确率
    test_label: 测试数据的真实标签, predicted_label: 预测的标签, class: 类别的数量
"""
def compute_per_class_acc(test_label, predicted_label, classes):
    test_label = np.array(test_label)
    predicted_label = np.array(predicted_label)
    acc_per_class = []

    # 总体准确率: test_label = predicted_label的数量，除以test_label的长度
    acc = np.sum(test_label == predicted_label) / len(test_label)

    # 计算当前类别的准确率: 计算在test_label和predicted_label中相等且在当前类别中的元素数量，并除以在当前类别中的元素数量
    for i in range(len(classes)):
        idx = (test_label == i)
        acc_per_class.append(np.sum(test_label[idx] == predicted_label[idx]) / np.sum(idx))
    return acc, sum(acc_per_class)/len(acc_per_class)


# def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
#         acc_per_class = []

#         acc = np.sum(test_label == predicted_label) / len(test_label)
#         for i in target_classes:
#             idx = (test_label == i)
#             acc_per_class.append(np.sum(test_label[idx] == predicted_label[idx]) / np.sum(idx))
#         return acc, sum(acc_per_class)/len(acc_per_class)


def calibrated_stacking(opt, output, lam=1e-3):
    """
    output: the output predicted score of size batchsize * 200
    lam: the parameter to control the output score of seen classes.
    self.test_seen_label
    self.test_unseen_label
    :return
    """
    output = output.cpu().numpy()
    seen_L = list(set(opt.test_seen_label.numpy()))
    output[:, seen_L] = output[:, seen_L] - lam
    return torch.from_numpy(output)


# def compute_accuracy(output, target, topk=(1, )):
#     """Computes the accuracy over the k top predictions for
#     the specified values of k.

#     Args:
#         output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
#         target (torch.LongTensor): ground truth labels with shape (batch_size).
#         topk (tuple, optional): accuracy at top-k will be computed. For example,
#             topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

#     Returns:
#         list: accuracy at top-k.
#     """
#     maxk = max(topk)
#     batch_size = target.size(0)

#     if isinstance(output, (tuple, list)):
#         output = output[0]

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#         acc = correct_k.mul_(100.0 / batch_size)
#         res.append(acc)

#     return res


# def compute_per_class_acc
def compute_acc_avg_per_class(test_label, predicted_label, nclass):
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    for i in range(nclass):
        idx = (test_label == i)
        if torch.sum(idx).float() != 0:
            acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
    return acc_per_class.mean()


# get the accuracy of each class
def compute_each_class_acc(test_label, predicted_label, nclass):
    test_label = torch.tensor(test_label)
    predicted_label = torch.tensor(predicted_label)

    acc_per_class = torch.FloatTensor(len(nclass)).fill_(0)
    for i in range(len(nclass)):
        idx = (test_label == i)
        idx = torch.tensor(idx)

        if torch.sum(idx).float() != 0:
            acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
    return acc_per_class


 # def compute_per_class_acc_gzsl
def compute_acc_avg_per_class_gzsl(self, test_label, predicted_label, nclass, cls_type):
    acc_per_class = 0
    if cls_type == 'seen':
        n = 0
    if cls_type == 'unseen':
        n = self.seenclasses.size(0)
    # print("n: ", n)
    for i in range(nclass):
        i = i + n
        idx = (test_label == i)
        if torch.sum(idx).float() == 0:
            continue
        else:
            acc_per_class += torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
    acc_per_class /= nclass
    return acc_per_class
    # compute Macro metric, i.e., average the accuracy of each class



def val_zsl(self, test_X, test_label, target_classes, second=False):
    start = 0
    ntest = test_X.size()[0]
    predicted_label = torch.LongTensor(test_label.size())
    # all_output = None
    for i in range(0, ntest, self.batch_size):
        end = min(ntest, start + self.batch_size)
        if self.cuda:
            output = self.model(Variable(test_X[start:end].cuda(), volatile=True))
        else:
            output = self.model(Variable(test_X[start:end], volatile=True))
        # if all_output is None:
        #     all_output = output
        # else:
        #     all_output = torch.cat((all_output, output), 0)
        _, predicted_label[start:end] = torch.max(output.data, 1)
        start = end
    overall_acc = self.compute_acc_avg_per_class(map_label(test_label, target_classes), predicted_label,
                                                    target_classes.size(0))
    acc_of_all = self.compute_each_class_acc(map_label(test_label, target_classes), predicted_label,
                                                target_classes.size(0))
    # return overall_acc, predicted_label, all_output, acc_of_all
    return overall_acc, acc_of_all


def val_gzsl(self, test_X, test_label, all_classes, target_classes, cls_type):
    start = 0
    ntest = test_X.size()[0]
    predicted_label = torch.LongTensor(test_label.size())
    all_output = None
    for i in range(0, ntest, self.batch_size):
        end = min(ntest, start + self.batch_size)
        if self.cuda:
            output = self.model(Variable(test_X[start:end].cuda(), volatile=True))
        else:
            output = self.model(Variable(test_X[start:end], volatile=True))

        # if all_output is None:
        #     all_output = output
        # else:
        #     all_output = torch.cat((all_output, output), 0)
        _, predicted_label[start:end] = torch.max(output.data, 1)
        start = end
    # acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label, target_classes.size(0))
    # overall_acc = self.compute_acc_avg_per_class_gzsl(test_label, predicted_label, target_classes)
    overall_acc = self.compute_acc_avg_per_class_gzsl(map_label(test_label, all_classes), predicted_label, target_classes.size(0), cls_type)
    return overall_acc, predicted_label



# classes: [image paths]
def calculate_accuracy(predictions, ground_truth):
    # print(f"predictions: {predictions.keys()}")

    class_accuracies = []
    error_class = []

    for class_name, img_list in predictions.items():
        if class_name not in ground_truth.keys():
            error_class.append(class_name)
            continue
        truth = ground_truth[class_name]
        if len(img_list) == 0 or len(truth) == 0 :
            continue

        correct_predictions = 0
        for img in img_list:
            if img in truth:
                correct_predictions += 1

        class_accuracy = float(correct_predictions) / min(len(img_list), len(truth))
        class_accuracies.append(class_accuracy)
        # print(f"class name: {class_name}, imgs: {img_list}, ground_truth:")
        # for e in ground_truth[class_name]:
        #     print(e) 
        # print(f"accuracy:{class_accuracy}")

    avg_acc = np.mean(class_accuracies) * 100
    # print(f"class_accuracies: {class_accuracies}, avg: {avg_acc}")
    
    return class_accuracies, avg_acc, error_class


# Hits@k反映了模型在前k个结果中的准确率
# 预测出的结果中，正确结果出现的平均倒数位置，MRR越大，模型的排序效果越好
# Hits@k衡量了模型的准确率，MRR衡量了模型的排序能力
def calculate_hits_mrr(ground_truths, predictions, k_values = [1, 3, 5, 10], task=None):
    print(f"Testing task is {task}")
    hits = {k: 0 for k in k_values}
    mrr = 0
    
    # print(f"ground truth: {len(ground_truths)}")
    # print(f"predict: {len(predictions)}, {sum(len(lst) for lst in predictions.values())}")

    for key in predictions.keys():

        if key not in ground_truths.keys():
            continue
        ground_truth = ground_truths[key]
        prediction = predictions[key]
        
        for k in k_values:
            rank = -1
            top_k_predictions = prediction[:k]

            if any(item in ground_truth for item in top_k_predictions):
                hits[k] += 1

                for i, item in enumerate(top_k_predictions):
                    if item in ground_truth:
                        rank = i + 1
                        break
                
                mrr += 1 / rank
    
    if task not in ["CUB_200", "AWA2", "SUN"]:
        total_keys = len(ground_truths.keys() & predictions.keys())
    else:
        total_keys = len(predictions)
    hits_at_k = {k: round(hits[k] / total_keys * 100.0, 5) for k in k_values}
    mean_reciprocal_rank = round(mrr / total_keys / len(hits), 5)
    
    # hits_at_k = {k: round(hits[k] / len(predictions) * 100.0, 5) for k in k_values}
    # mean_reciprocal_rank = round(mrr / len(predictions) / len(hits), 5)
    
    return hits_at_k, mean_reciprocal_rank



def retrieve_images(predicted, imgs, classes, epoch_pred):
    """
        Retrieve images for text
    """
    # logits = (100.0 * text_features @ image_features.T).softmax(dim=-1)

    pred_class_img = {}

    for i in range(len(classes)):  # class
        top_k_imgs = [imgs[index] for index in predicted[i]] # image
        for im in top_k_imgs:
            if classes[i] not in pred_class_img.keys():
                pred_class_img[classes[i]] = []
            pred_class_img[classes[i]].append(im)

            if classes[i] not in epoch_pred.keys():
                epoch_pred[classes[i]] = []
            epoch_pred[classes[i]].append(im)

    return pred_class_img, epoch_pred


def image_classification(predicted, imgs, classes, epoch_pred):
    """
        Retrieve texts for image: k = 1 (image classification)
    """
    # logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    pred_class_img = {}

    for i in range(len(imgs)):  # image
        top_k_txts = [classes[index] for index in predicted[i]] # class
        for cl in top_k_txts:
            if cl not in pred_class_img.keys():
                pred_class_img[cl] = []
            pred_class_img[cl].append(imgs[i])

            if cl not in epoch_pred.keys():
                epoch_pred[cl] = []
            epoch_pred[cl].append(imgs[i])


    return pred_class_img, epoch_pred

