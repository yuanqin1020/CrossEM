
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from argparse import ArgumentParser

import requests
from PIL import Image
import json
import clip
import time

from utils import *
from match import *
    


def batch_guidance(dict, img_paths):
    guidances = []
    for img in img_paths:
        guidances.append(dict[img])

    return guidances

def read_guidance(file_path):
    dict = {}  
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)  
            image_path = data['image_path']
            captions = data['caption']
            scores = data['scores']

            max_score = max(scores)
            max_index = scores.index(max_score)
            max_caption = captions[max_index]

            dict[image_path] = max_caption
            
    print(f"Number of guidances in cache: {len(dict)}")

    return dict



def encode_text(self, text):
    x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

    x = x + self.positional_embedding.type(self.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.ln_final(x).type(self.dtype)

    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

    return x

class GuidedCustomCLIP(nn.Module):
    
    def __init__(self, args, clip_model, combine, fusion, device, batch_size):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.ctx_bais = args.ctx_bais
        self.device = device
        self.prompt_learner = MatchPromptLearner(args.p_clusters, clip_model, args.ctx_init, 
                                                 self.ctx_bais, args.n_ctx, args.tk_w, args.device, args.clss)
        
        self.embed_fn = clip_model.token_embedding
        self.vis_dim = clip_model.visual.output_dim
        self.txt_dim = clip_model.ln_final.weight.shape[0]
        self.clip_tencoder = clip_model.encode_text
        self.batch_size = batch_size

        self.combine = combine
        self.fusion = fusion
        self.fusion_reduction = nn.Linear(self.txt_dim + self.vis_dim, self.vis_dim).half()
        
        if self.combine == "features":
            self.merged_feature_head = nn.Sequential(nn.Linear(4 * self.batch_size, self.vis_dim), 
                                                     nn.LeakyReLU()).half() # nn.ModuleList()

        
    def forward(self, images, classnames, clembs, device, guidances=None):
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(images.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts, tokenized_prompts = self.prompt_learner(image_features, classnames, clembs, device)

        text_features = []
        for pts_i in prompts:  # (1, n_cls, n_tkn, tx_dim)
            text_feature = self.text_encoder(pts_i, tokenized_prompts)

            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            text_features.append(text_feature)

        text_features = torch.stack(text_features) 

        logits = logit_scale * text_features @ image_features.t()
        logits = logits.squeeze()

        if guidances != None:
            tokenized_giduance = torch.cat([clip.tokenize(g) for g in guidances]).to(self.device)  # (n_guide, n_tkn=77)
            with torch.no_grad():
                guide_features = self.clip_tencoder(tokenized_giduance).type(self.dtype)
            guide_features = guide_features / guide_features.norm(dim=-1, keepdim=True)
            guide_features = guide_features.view(guide_features.size(0), -1)

            guide_logits = logit_scale * text_features @ guide_features.t()
            guide_logits = guide_logits.squeeze()

        if self.combine == "logits":
            simi_logits = logits   
        elif self.combine == "guidance":
            simi_logits = guide_logits
        elif self.combine == "features":
            fusion_feat = torch.cat((guide_features, image_features), dim=1).half()
            fusion_feat = self.fusion_reduction(fusion_feat) 
            fuse_logits = logit_scale * text_features @ fusion_feat.t()
            fuse_logits = fuse_logits.squeeze()
            print(f"{logits.shape}, {guide_logits.shape}, {fuse_logits.shape}")

            simi_logits = fuse_logits
        
        return simi_logits, tokenized_prompts

class GuidedDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        # self.guidances = guidances
        self.images = image_paths
        self.transform = Compose([Resize((224, 224), interpolation=Image.BICUBIC), CenterCrop(224), 
                                  lambda image: image.convert("RGB"), ToTensor(), 
                                  Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        image = Image.open(img_path)
        image = self.transform(image)

        return image, img_path
    

def generate_batches(classnames, train_imgs, cluster_num, batch_size, classes_images_simi, c_prune, hns, theta_1, theta_2, bl):
    dataloaders = []
    if bl in ['pcp', 'hash', "simi"]:
        images_list = block_images(classes_images_simi, classnames, train_imgs, cluster_num, batch_size, c_prune, hns, theta_1, theta_2, bl)
        for ind in range(len(images_list)):
            imgs = images_list[ind]
            if len(imgs) <= 1: continue

            dataset = GuidedDataset(imgs)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            dataloaders.append(dataloader)

    else:
        dataset = GuidedDataset(train_imgs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloaders.append(dataloader)

    return dataloaders



def guided_train(args, train_sets, train_imgs, k_values, true_class_img, classes_images_simi, train_data, label_freq, label_neighbors):
    task = args.data.strip().split("/")[-2]
    
    clip_model = load_clip_to_cpu(args.backbone)
    torch.manual_seed(args.seed)
    
    model = GuidedCustomCLIP(args, clip_model, args.combine, args.fusion, args.device, args.p_batch_size)
    model = model.to(args.device)
    print("Turning off gradients in both the image and the text encoder")
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
            print(f"{name}: {param.data.shape}")
    print(f"Parameters to be updated: {enabled}")

    for param in model.parameters():
        param.to(args.device)

    optimizer = optim.Adam(model.prompt_learner.parameters(), lr=args.p_lr) 

    for epoch in range(args.p_epochs):

        total_loss = 0
        epoch_pred = {}
        batch = 0
        total_batch = 0
        
        ep_time = []
        runtime = 0

        for ind in range(len(train_sets)): 
            classnames = list(train_sets[ind].keys())
            if len(classnames) <= 1: continue

            embs = list(train_sets[ind].values())        
            model.prompt_learner.initialize_prompts(classnames, train_data, label_freq, label_neighbors)

            dataloaders = generate_batches(classnames, train_imgs, args.cluster_num, args.p_batch_size, 
                                           classes_images_simi, args.c_prune, args.hns, args.theta_1, args.theta_2, args.bl)
            start = time.time()

            batch_num = sum([len(dataloader) for dataloader in dataloaders])
            total_batch += batch_num
            print(f"Number of train batches in block {ind+1}/{len(train_sets)}: {batch_num}")

            for dataloader in dataloaders:   # batches
                optimizer.zero_grad()
                
                batch_loss = 0
                mbatch = 0
                # batch_logits = torch.tensor([]).to(args.device)
                # batch_path = []
                bimg_num = sum(len(imgs) for imgs, _ in dataloader)
                print(f"Number of classes and images in batch {batch+1}: {len(classnames)}, {bimg_num}")

                for imgs, paths in dataloader:
                    if len(imgs) <= 1: continue
                    print(f"Number of classes and images in minibatch {mbatch+1}/{len(dataloader)}: {len(classnames)}, {len(imgs)}")

                    imgs = imgs.to(args.device)
                    logits, tokenized_prompts = model(imgs, classnames, embs, args.device, guidances=None) 
                    loss = compute_loss(logits, tokenized_prompts, args.oloss)
                
                    torch.autograd.set_detect_anomaly(True)
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    
                    if logits.dim() == 1:
                        logits = torch.unsqueeze(logits, dim=0)
                    _, predicted = torch.topk(logits, 1, dim=1) 
                    pred_class_img, epoch_pred = retrieve_images(predicted, paths, classnames, epoch_pred)

                    batch_loss += loss.item()
                    mbatch += 1
                    hits_at_k, mrr = calculate_hits_mrr(true_class_img, pred_class_img, k_values, task)
                    print(f"{epoch+1}/{args.p_epochs} mini batch {mbatch} loss: {loss.item()}, hits@k: {hits_at_k}, mrr: {mrr}")
            
                batch += 1
                total_loss += batch_loss
            
            runtime += time.time() - start

        average_loss = total_loss / total_batch
        print(f"Number of total batches in epoches: {total_batch}")
        hits_at_k, mrr = calculate_hits_mrr(true_class_img, epoch_pred, k_values, task)
        
        ep_time.append(runtime)
        print(f"Epoch: {datetime.datetime.now()} - {epoch+1}/{args.p_epochs}, Loss: {average_loss}: hits@k: {hits_at_k}, mrr: {mrr}, time: {runtime} Sec")
   

    print(f"average runtime in epoches: {round(sum(ep_time) / len(ep_time), 5)} Sec")

    torch.save(model.state_dict(), get_model_path(args).replace(".pt", "_" + args.combine + ".pt")) 
    print("Guided model saved.")



def guided_test(args, test_imgs, test_emb, k_values, true_class_img, test_data, label_freq, label_neighbors):
    clip_model = load_clip_to_cpu(args.backbone)
    model_path = get_model_path(args).replace(".pt", "_" + args.combine + ".pt")
    print(f"Path of test model: {model_path}")

    model = GuidedCustomCLIP(args, clip_model, args.combine, args.fusion, args.device, args.p_batch_size).to(args.device)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    dataset = GuidedDataset(test_imgs)
    dataloader = DataLoader(dataset, batch_size=args.p_batch_size, shuffle=False)
    print(f"Number of test batches: {len(dataloader)}")

    test_data = test_data.to(args.device)

    classnames = list(test_emb.keys())
    if args.clss == "feat":
        vers = torch.stack(list(test_emb.values())).to(args.device)
        class_emb = model.prompt_learner.feat_reduction(vers) 
    elif args.clss == "freq":
        edge_freq = edge_prune_probability(test_data, label_neighbors, label_freq)
        class_emb = model.prompt_learner.prune_gat(test_data, edge_freq, classnames, args.device).to(args.device)
    else:
        print(f"test using {args.clss}")
        class_emb = list(test_emb.values())

    model.prompt_learner.initialize_prompts(classnames, test_data, label_freq, label_neighbors)

    similarity = []
    pred_class_img = {}
    with torch.no_grad():
        for batch_img, batch_path in dataloader:
            batch_img = batch_img.to(args.device)
            logits, _ = model(batch_img, classnames, class_emb, args.device) # image-text
            similarity.append(logits)
        
        similarity = torch.cat(similarity, dim=1)

        _, predicted = torch.topk(similarity, args.tops, dim=1)  
        _, pred_class_img = retrieve_images(predicted, test_imgs, classnames, pred_class_img)

        task = args.data.strip().split("/")[-2]
        hits_at_k, mrr = calculate_hits_mrr(true_class_img, pred_class_img, k_values, task)
    
    print(f"Test hits@k: {hits_at_k}, mrr: {mrr}")


def guided_pipline(args, train_img, train_sets, test_unseen_img, test_unseen_embs,
                   dataset, k_values, classes_images_simi, train_data, data, label_freq, label_neighbors):
    if args.p_mode =="train":
        true_class_img = true_targets(train_img, args.data)
        
        guided_train(args, train_sets, train_img, k_values, true_class_img, classes_images_simi, train_data, label_freq, label_neighbors)


    test_img = test_unseen_img
    test_emb = {**test_unseen_embs}
    true_class_img = true_targets(test_img, args.data)
    
    guided_test(args, test_img, test_emb, k_values, true_class_img, data, label_freq, label_neighbors)

