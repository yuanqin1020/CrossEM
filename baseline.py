
import clip
import torch
import time
from prompt import CustomDataset
from evaluate import true_targets, calculate_hits_mrr, map_img
from torch_geometric.data import DataLoader, Data

from loader import attr_encoder, bert_emd
from guidance import read_guidance
from utils import image_patches
from graph_model import init_ver_emd, read_graph_data
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from utils import gen_vertex_text, gen_hard_prompt


k_values = [1, 3, 5, 10]

def hard_prompt_method(args, classes, imgs, dataset, entity2text=None):
    start = time.time()

    if dataset in ["CUB_200", "AWA2", "SUN"]:
        file = 'new_triplets.txt' if args.new_trip != "" else 'triplets.txt'
    else:
        file = "test_triplets.txt"
    file_path = args.root + args.data + file
    vertex_attrs, edge_attrs, edges, edge_attr = read_graph_data(file_path)
    model, tokenizer = attr_encoder(args.root + args.bert_name)
    data = init_ver_emd(vertex_attrs, edges, edge_attr, edge_attrs, model, tokenizer)

    dict, _ = gen_vertex_text(data, edges)
    prompts = gen_hard_prompt(dict)

    model, _ = clip.load("ViT-B/32", args.device)
    text_input = torch.cat([clip.tokenize(str(prompts[c])[:77]) for c in classes]).to(args.device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    true_class_img = true_targets(imgs, args.data, entity2text)
    # print(f"true_class_img: {true_class_img}")

    dataset = CustomDataset(imgs)
    dataloader = DataLoader(dataset, batch_size=30, shuffle=False)
    print(f"Number of test batches: {len(dataloader)}")

    image_features = []
    with torch.no_grad():
        for batch_img, batch_path in dataloader:
            batch_img = batch_img.to(args.device)
            features = model.encode_image(batch_img.to(args.device))
            # print(f"features:{features.shape}")
            image_features.append(features)

        image_features = torch.cat(image_features)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    task = args.data.strip().split("/")[-2]
    _ = retrieve_images(image_features, text_features, imgs, classes, true_class_img, args.tops, task)

    image_classification(image_features, text_features, imgs, classes, true_class_img, tops=1)

    print(f"taking time: {time.time() - start} Sec")



def base_method(args, classes, imgs, entity2text=None):
    start = time.time()
    model, preprocess = clip.load("ViT-B/32", args.device)
    text_input = torch.cat([clip.tokenize(str(c))[:77] for c in classes]).to(args.device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    true_class_img = true_targets(imgs, args.data, entity2text)
    # print(f"true_class_img: {true_class_img}")

    dataset = CustomDataset(imgs)
    dataloader = DataLoader(dataset, batch_size=30, shuffle=False)
    print(f"Number of test batches: {len(dataloader)}")

    image_features = []
    with torch.no_grad():
        for batch_img, batch_path in dataloader:
            batch_img = batch_img.to(args.device)
            features = model.encode_image(batch_img.to(args.device))
            # print(f"features:{features.shape}")
            image_features.append(features)

        image_features = torch.cat(image_features)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    task = args.data.strip().split("/")[-2]
    _ = retrieve_images(image_features, text_features, imgs, classes, true_class_img, args.tops, task)

    image_classification(image_features, text_features, imgs, classes, true_class_img, tops=1)

    print(f"taking time: {time.time() - start} Sec")

# sentence_method(args, test_unseen_classes, dataset, test_unseen_img, "test_unseen")
def sentence_method(args, classes, dataset, images):
    start = time.time()

    if dataset in ["CUB_200", "AWA2", "SUN"]:
        file = 'new_triplets.txt' if args.new_trip != "" else 'triplets.txt'
        mode = "test_unseen"
    else:
        file = "test_triplets.txt"
        mode = "test"
    file_path = args.root + args.data + file
    vertex_attrs, edge_attrs, edges, edge_attr = read_graph_data(file_path)
    model, tokenizer = attr_encoder(args.root + args.bert_name)
    data = init_ver_emd(vertex_attrs, edges, edge_attr, edge_attrs, model, tokenizer)

    dict, _ = gen_vertex_text(data, edges)
    prompts = gen_hard_prompt(dict)
    prompt_feats = bert_emd(prompts, model, tokenizer)

    guidances = read_guidance(args.root + "cache/guidance/" + mode + "_" + dataset + "_with_guidance.json")
    caption_feats = bert_emd(list(guidances.values()), model, tokenizer)
    true_class_img = true_targets(images, args.data)

    _ = retrieve_images(caption_feats, prompt_feats, list(guidances.keys()), classes, true_class_img, args.tops, dataset)

    print(f"taking time: {time.time() - start} Sec")


def patches_method(args, images, device, dataset, mode, new_trip, file):
    print(f"Number of patch preprocessing {mode}: {len(images)}")
    file_path = args.root + args.data + file
    vertex_attrs, edge_attrs, edges, edge_attr = read_graph_data(file_path)
    model, tokenizer = attr_encoder(args.root + args.bert_name)

    data = init_ver_emd(vertex_attrs, edges, edge_attr, edge_attrs, model, tokenizer)
    attribute_feats = data.x

    # 加载预训练的ResNet模型
    resnet = resnet18(pretrained=True)
    resnet.eval()
    fc = torch.nn.Linear(1000, 768)

    # 图片转换为tensor并进行预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图片大小
        transforms.ToTensor(),  # 转换为tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # img_patches = image_patches(images, device)
    # patches = list(img_patches.values())

    img_attr = torch.zeros(len(images), len(data.x))
    suff = "_new.txt" if new_trip != "" else ".txt"
    with open("./cache/feat/logits_preprocess_" + dataset + "_" + mode + suff, 'a') as f:
        for i in range(len(images)):
            # if i < 9012: continue 
            img = images[i]
            print(f"Patches: {img}")
            try:
                patches = image_patches(img, transform, resnet, fc)
                
                for j in range(data.x.size(0)):
                    logits = (100.0 * attribute_feats[j] @ patches.T).softmax(dim=-1) # (attr, patches)
                    img_attr[i, j] = torch.max(logits)
                    pair = {"image": img,
                            "attribute": data.y[j],
                            "sum": torch.sum(logits).detach().item(),
                            "avg": torch.mean(logits).detach().item(),
                            "max": torch.max(logits).detach().item()
                    }
                    # json.dump(pair, f, ensure_ascii=False)
                    f.write(str(pair) + "\n")

            except Exception as e:
                print(f"Error occurred while processing image {img}: {e}")
                continue

                # if pair["attribute"] in ["black footed albatross", "solid", "black", "about the same as head", "medium (9 - 16 in)"]:
                #     print(pair["attribute"])
                #     _, ind = torch.max(logits, dim=0)
                #     img_data = patches[ind].detach().numpy()
                #     img_shape = img_data.shape
                #     if len(img_shape) == 3:
                #         plt.imshow(img_data)
                #     elif len(img_shape) == 2:
                #         plt.imshow(img_data, cmap='gray')
                #     plt.axis('off')
                #     plt.show()

            # f.write(json.dumps(preprocess, ensure_ascii=False).encode('utf-8'))
            # json.dump(preprocess, f, ensure_ascii=False)

            if i % 100 == 0 or i == len(images) - 1:
                print(f"preprocess the patches of {i}-th image")
            
    print("Preprocess patches dump finish.")

    return img_attr


def patches_simi_cache(args, images, device, dataset, mode, new_trip, file):
    print(f"Number of patch preprocessing {mode}: {len(images)}")
    file_path = args.root + args.data + file
    vertex_attrs, edge_attrs, edges, edge_attr = read_graph_data(file_path)
    model, tokenizer = attr_encoder(args.root + args.bert_name)

    data = init_ver_emd(vertex_attrs, edges, edge_attr, edge_attrs, model, tokenizer)
    attribute_feats = data.x

    # 加载预训练的ResNet模型
    resnet = resnet18(pretrained=True)
    resnet.eval()
    fc = torch.nn.Linear(1000, 768)

    # 图片转换为tensor并进行预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图片大小
        transforms.ToTensor(),  # 转换为tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # img_patches = image_patches(images, device)
    # patches = list(img_patches.values())
    img_attr = torch.zeros(len(images), len(data.x))
    suff = "_new.txt" if new_trip != "" else ".txt"
    with open("./cache/feat/logits_preprocess_" + dataset + "_" + mode + suff, 'a') as f:
        for i in range(len(images)):
            # if i < 9012: continue 
            img = images[i]
            print(f"Patches: {img}")
            try:
                patches = image_patches(img, transform, resnet, fc)
                
                for j in range(data.x.size(0)):
                    logits = (100.0 * attribute_feats[j] @ patches.T).softmax(dim=-1) # (attr, patches)
                    img_attr[i, j] = torch.max(logits)
                    attribute = data.y[j]
                    max_value = round(torch.max(logits).detach().item(), 5)
                    img_str = img.replace(args.root, "")
                    f.write(img_str + ", " + attribute + ", " + str(max_value) + "\n")

                del patches

            except Exception as e:
                print(f"Error occurred while processing image {img}: {e}")
                continue

            if i % 100 == 0 or i == len(images) - 1:
                print(f"preprocess the patches of {i}-th image")
            
    print("Preprocess patches dump finish.")

    return img_attr


def comparing(root, bert_name, images):
    model, tokenizer = attr_encoder(root + bert_name)
    sentences = ["laysan albatross has underparts color in white", "laysan albatross has eye color in black"]
    sent_features = bert_emd(sentences, model, tokenizer)

    images = ["/home/yuanqin/Align/datasets/img_cf/CUB_200/images/065.Slaty_backed_Gull/Slaty_Backed_Gull_0046_796035.jpg", 
              "/home/yuanqin/Align/datasets/img_cf/CUB_200/images/053.Western_Grebe/Western_Grebe_0022_36148.jpg",
              "/home/yuanqin/Align/datasets/img_cf/CUB_200/images/025.Pelagic_Cormorant/Pelagic_Cormorant_0053_23760.jpg", 
              "/home/yuanqin/Align/datasets/img_cf/CUB_200/images/001.Black_footed_Albatross/Black_Footed_Albatross_0039_796132.jpg",
              "/home/yuanqin/Align/datasets/img_cf/CUB_200/images/101.White_Pelican/White_Pelican_0048_95764.jpg",
              "/home/yuanqin/Align/datasets/img_cf/CUB_200/images/002.Laysan_Albatross/Laysan_Albatross_0001_545.jpg",
              "/home/yuanqin/Align/datasets/img_cf/CUB_200/images/002.Laysan_Albatross/Laysan_Albatross_0002_1027.jpg",
              "/home/yuanqin/Align/datasets/img_cf/CUB_200/images/002.Laysan_Albatross/Laysan_Albatross_0100_735.jpg",
              "/home/yuanqin/Align/datasets/img_cf/CUB_200/images/002.Laysan_Albatross/Laysan_Albatross_0017_614.jpg",
              "/home/yuanqin/Align/datasets/img_cf/CUB_200/images/002.Laysan_Albatross/Laysan_Albatross_0003_1033.jpg"]

    # 加载预训练的ResNet模型
    resnet = resnet18(pretrained=True)
    resnet.eval()
    fc = torch.nn.Linear(1000, 768)

    # 图片转换为tensor并进行预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图片大小
        transforms.ToTensor(),  # 转换为tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    img_attr = torch.zeros(len(images), len(sentences))
    for i in range(len(images)):
        img = images[i]
        print(f"Patches: {img}")
        try:
            patches = image_patches(img, transform, resnet, fc)
            
            for j in range(len(sentences)):
                logits = (100.0 * sent_features[j] @ patches.T).softmax(dim=-1) # (attr, patches)
                img_attr[i, j] = torch.max(logits)
                pair = {"image": img,
                        "attribute": sentences[j],
                        "sum": torch.sum(logits).detach().item(),
                        "avg": torch.mean(logits).detach().item(),
                        "max": torch.max(logits).detach().item()
                }
                # json.dump(pair, f, ensure_ascii=False)
                print(str(pair))

        except Exception as e:
            print(f"Error occurred while processing image {img}: {e}")
            continue







def retrieve_images(image_features, text_features, imgs, classes, true_class_img, tops, task):
    """
        Retrieve images for text
    """
    logits = (100.0 * text_features @ image_features.T).softmax(dim=-1)

    pred_class_img = {}
    _, predicted = torch.topk(logits, tops, dim=1) 
    for i in range(len(classes)):  # class
        top_k_imgs = [imgs[index] for index in predicted[i]] # image
        for im in top_k_imgs:
            if classes[i] not in pred_class_img.keys():
                pred_class_img[classes[i]] = []
            pred_class_img[classes[i]].append(im)
            
    print(f"pred_class_img: {len(pred_class_img)}")
    hits_at_k, mrr = calculate_hits_mrr(true_class_img, pred_class_img, k_values, task)
    
    print(f"Retrieve images for texts hits@k: {hits_at_k}, mrr: {mrr}")
    return pred_class_img


def image_classification(image_features, text_features, imgs, classes, true_class_img, tops=1):
    """
        Retrieve texts for image: k = 1 (image classification)
    """
    logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    pred_class_img = {}
    _, predicted = torch.topk(logits, tops, dim=1) 
    for i in range(len(imgs)):  # image
        top_k_txts = [classes[index] for index in predicted[i]] # class
        for cl in top_k_txts:
            if cl not in pred_class_img.keys():
                pred_class_img[cl] = []
            pred_class_img[cl].append(imgs[i])

    print(f"pred_class_img: {len(pred_class_img)}")
    hits_at_k, mrr = calculate_hits_mrr(true_class_img, pred_class_img, k_values)
    
    print(f"Image classification hits@k: {hits_at_k}, mrr: {mrr}")

