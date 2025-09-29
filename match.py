   

import argparse
import datetime
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from collections import OrderedDict
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.optim.lr_scheduler import ReduceLROnPlateau

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from loader import *
from utils import *
from evaluate import *
from graph_model import *


from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


# Define dataset class for vertices and images
class MatchCustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, visual):
        # self.text = list(ver_dict.keys())
        self.images = image_paths
        self.transform = Compose([Resize((224, 224), interpolation=Image.BICUBIC), CenterCrop(224), 
                                  lambda image: image.convert("RGB"), ToTensor(), 
                                  Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # print(f"idx: {idx}")
        img_path = self.images[idx]
        # label, _ = map_img(img_path)
        
        image = Image.open(img_path)
        image = self.transform(image)

        return image, img_path



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # print(f"prompt T: {prompts.shape}, token prompts: {tokenized_prompts.shape}")
        # 将prompts与位置嵌入相加，并将其转置为形状为(L, N, D)的张量，其中L是文本的长度，N是批量大小，D是嵌入的维度
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer(x)
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x[tokenized_prompts有最大值的索引在x中对应的元素]
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    


class MatchPromptLearner(nn.Module):
    """
        Unified context: to introduce M learnable context vectors, 
        so that the prompt for the i-th vertex is denoted by ti = {c1, c2, ..., cM, vi}
    """
    def __init__(self, num_clusters, clip_model, ctx_init, ctx_bais, n_ctx, tk_w, device, clss):
        super(MatchPromptLearner, self).__init__()
        self.num_clusters = num_clusters
        self.device = device
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.vis_dim = clip_model.visual.output_dim
        self.dtype = clip_model.dtype
        self.embed_fn = clip_model.token_embedding
        self.n_ctx = n_ctx
        self.ctx_bais = ctx_bais
        self.clss = clss
        self.ver_dim = 128

        if ctx_init == "init":
            ctx_init = "a photo of a"
            print("Use given words to initialize context prefix vectors")
            ctx_init = ctx_init.replace("_", " ")
            self.prompt_prefix = ctx_init
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = self.embed_fn(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]

        else:
            print("Use random context prefix initialization")
            self.prompt_prefix = " ".join(["X"] * self.n_ctx)
            ctx_vectors = torch.empty(n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
        
        print(f'Initial context: "{self.prompt_prefix}"')
        print(f"Number of context words (tokens): {self.n_ctx}")
            
        self.ctx = nn.Parameter(ctx_vectors)  # (n_ctx, ctx_dim)
        # print(f"self.ctx: {self.ctx}, {self.ctx.shape}")

        if self.ctx_bais == "cluster":
            self.cluster_reduction = nn.Sequential(nn.Linear(self.ver_dim, self.ctx_dim), nn.ReLU())
        elif self.ctx_bais == "meta":
            self.meta_net = nn.Sequential(nn.Linear(self.vis_dim, self.ctx_dim), nn.LeakyReLU())
            self.meta_net.half()
            # print(f"self.meta_net: {self.meta_net}")

        self.token_prefix = None
        self.token_suffix = None
        self.tokenized_prompts = None

        if self.clss == "feat":
            self.feat_reduction = nn.Sequential(nn.Linear(self.ver_dim, 77), nn.ReLU()).to(self.device)

        """ fixed init token weight """
        tk_weight_vector = torch.tensor([tk_w] * 77).to(self.device)
        # tk_weight_vector = torch.full((77,), tk_w)
        self.token_weight = nn.Parameter(tk_weight_vector)

        # label frequent
        if self.clss == "freq":
            self.prune_gat = PruneGAT(768, 77)
            # self.classes_feat = None
            # self.neighbors_feats = None
            # weight_vector = torch.full((self.ctx_dim,), 0.5, requires_grad=True).to(self.dtype)
            # self.weight = nn.Parameter(weight_vector)
    

    def initialize_prompts(self, classnames, data=None, label_freq=None, label_neighbors=None):
        """
        方法遍历每个classname, 为每个classname创建prompt。
        对于每个classname, 它将其前缀和后缀token化, 并使用CLIP模型的`token_embedding`方法将其转换为嵌入表示。
        然后, 它将前缀和后缀的嵌入表示存储在`token_prefix`和`token_suffix`列表中, 并将classname的prompt存储在`tokenized_prompts`列表中。
        最后, 它将每个classname的prompt的context部分存储在`ctx`列表中。
        """
        prompts = [self.prompt_prefix + " " + name for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p[:77]) for p in prompts]).to(self.device)  # (n_cls, n_tkn=77)
        
        with torch.no_grad():
            self.token_embed = self.embed_fn(tokenized_prompts).type(self.dtype) # (n_cls, n_tkn, ctx_dim)
        # print(f"self.token_embed: {self.token_embed.shape}")

        self.token_prefix = self.token_embed[:, :1, :].to(self.device)  # SOS
        self.token_suffix = self.token_embed[:, 1 + self.n_ctx :, :].to(self.device)   # CLS, EOS
        self.tokenized_prompts = tokenized_prompts.to(self.device)   # torch.Tensor
        # print(f"token_prefix - token_suffix: {self.token_prefix.shape}, {self.token_suffix.shape}")

        if label_freq is not None:
            self.re_data = reconstruct_data(data, classnames, self.device).to(self.device) 
            self.edge_freq = edge_prune_probability(self.re_data, label_neighbors, label_freq)


    def forward(self, im_features, classnames, clemb, device):
        prefix = self.token_prefix
        suffix = self.token_suffix
        tokenized_prompts = self.tokenized_prompts
        ctx = self.ctx.clone()                    # (n_ctx, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        # print(f"ctx base: {ctx.shape}") 

        """ tokenized_prompts和embedding(todo)中加入class feature信息 """
        if self.clss in ["feat", "freq"]:
            token_weight = self.token_weight.clone()
            token_weight = token_weight.unsqueeze(0)

            if self.clss == "feat":
                vers = torch.stack(clemb).to(self.device)
                class_emb = self.feat_reduction(vers) # (n_cls, n_tkn)
                tokenized_prompts = torch.cat([tokenized_prompts * (1 - token_weight) + class_emb * token_weight], dim=1)

            elif self.clss == "freq":
                class_emb = self.prune_gat(self.re_data, self.edge_freq, classnames, device).to(self.device)
                # print(f"tokenized_prompts: {tokenized_prompts.shape}, token_weight: {token_weight.shape}, class_emb: {class_emb.shape}")
                tokenized_prompts = torch.cat([tokenized_prompts * (1 - token_weight) + class_emb * token_weight], dim=1)

            self.tokenized_prompts = tokenized_prompts.to(self.device)   # torch.Tensor
            # print(f"feat tokenized_prompts: {self.tokenized_prompts.shape}")  
            
        if self.ctx_bais == "meta":
            print(f"Meta context bais")
            bias = self.meta_net(im_features)  # (batch, ctx_dim)
            bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
            print(f"meta bais: {bias}")

            # print(f"bais: {bias.shape}, ctx: {ctx.shape}")
            ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        
        else:
            ctx_shifted = ctx
        # print(f"ctx shifted: {ctx_shifted.shape}")

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(len(classnames), -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
            # print(f"ctx_shifted_i : {ctx_shifted_i}")

        prompts = torch.stack(prompts)  # (1, n_cls, n_tkn, ctx_dim)
        # print(f"prompts end shape: {prompts.shape}")
        
        return prompts, tokenized_prompts
    

    def construct_prompts(self, ctx, prefix, suffix):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts.type(self.dtype)
    


class MatchCustomCLIP(nn.Module):
    
    def __init__(self, args, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.ctx_bais = args.ctx_bais

        self.prompt_learner = MatchPromptLearner(args.p_clusters, clip_model, args.ctx_init, 
                                            self.ctx_bais, args.n_ctx, args.tk_w, args.device, args.clss)
        
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
    def forward(self, image, device, classnames, clemb, task, label=None):
        # tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # print(f"image_features: {image_features.shape}")

        prompts, tokenized_prompts = self.prompt_learner(image_features, classnames, clemb)
        print(f"prompt C: {prompts}")
        print(f"token prompts: {tokenized_prompts.shape}")

        logits = []
        if self.ctx_bais == "meta":
            for pts_i, imf_i in zip(prompts, image_features):
                text_features = self.text_encoder(pts_i, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                # logits_per_image = logit_scale * image_features @ text_features.t()
                # logits_per_text = logit_scale * text_features @ image_features.t()

                if task == "imgcf":
                    l_i = logit_scale * imf_i @ text_features.t()
                else: # kgcom
                    l_i = logit_scale * text_features @ imf_i.t()
                logits.append(l_i)

            # print(f"logits 1: {len(logits)}, {len(logits[0])}")
            logits = torch.stack(logits)

        else:
            # unified context
            for pts_i in prompts:
                text_features = self.text_encoder(pts_i, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                if task == "imgcf":
                    logits.append(logit_scale * image_features @ text_features.t())
                else: # kgcom
                    logits.append(logit_scale * text_features @ image_features.t())

            logits = torch.stack(logits).squeeze()
        
        return logits

   
def match_train(args, train_sets, train_imgs, k_values, true_class_img, label_freq, label_neighbors):
    clip_model = load_clip_to_cpu(args.backbone)
    torch.manual_seed(args.seed)
    
    # train_class = list(train_embeds.keys())
    model = MatchCustomCLIP(args, clip_model)
    model = model.to(args.device)
    # print(f"model: {model}")

    print("Turning off gradients in both the image and the text encoder")
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    # model.logit_scale.requires_grad = False


    # Double check
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")

    optimizer = optim.Adam(model.prompt_learner.parameters(), lr=args.p_lr) # Only optimize PromptLearner parameters
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

    dataset = MatchCustomDataset(train_imgs, clip_model.visual)
    dataloader = DataLoader(dataset, batch_size=args.p_batch_size, shuffle=True)
    print(f"Number of train batches: {len(dataloader)}")

    criterion = nn.CrossEntropyLoss()
    # class_index = {string: i for i, string in enumerate(train_classes)}

    for epoch in range(args.p_epochs):
        # if epoch > 1: break

        total_loss = 0
        batch = 0
        epoch_pred = {}
        
        for batch_img, batch_path in dataloader:
            # if batch > 5: break 

            batch_logits = torch.tensor([]).to(args.device)
            block_loss = 0

            for ind in range(len(train_sets)):
                classnames = list(train_sets[ind].keys())
                embs = list(train_sets[ind].values())
                # print(f"Block-{ind} classes number: {len(classnames)}")

                model.prompt_learner.initialize_prompts(classnames, label_freq, label_neighbors)
                optimizer.zero_grad()
                batch_img = batch_img.to(args.device)

                # 将模型参数移动到设备上
                for param in model.parameters():
                    param.to(args.device)

                logits = model(batch_img, args.device, classnames, embs, args.task) # (batch image, block size)
                batch_logits = torch.cat((batch_logits, logits), dim=1)

            _, predicted = torch.topk(batch_logits, 1, dim=1) # pos: top 1

            if args.task == "kgcom":
                pred_class_img, epoch_pred = retrieve_images(predicted, batch_path, classnames, epoch_pred)
            else:
                pred_class_img, epoch_pred = image_classification(predicted, batch_path, classnames, epoch_pred)

        
            # print(f"batch_logits: {batch_logits.shape}")
            loss = constractive_loss(batch_logits)  # (batch image, ver size)
            # Backward pass and optimization
            loss.backward(retain_graph=True)
            optimizer.step()

            block_loss += loss.item()
            
        
            hits_at_k, mrr = calculate_hits_mrr(true_class_img, pred_class_img, k_values)
            print(f"{epoch+1}/{args.p_epochs} batch {batch+1}/{len(dataloader)} loss: {block_loss}, hits@k: {hits_at_k}, mrr: {mrr}")
        
            batch += 1
            total_loss += block_loss

        # # 计算epoch loss和accuracy
        # _, accuracy, bug_classes = calculate_accuracy(epoch_pred, true_class_img)

        hits_at_k, mrr = calculate_hits_mrr(true_class_img, epoch_pred, k_values)
        average_loss = total_loss / len(dataloader)
        print(f"Epoch: {datetime.datetime.now()} - {epoch+1}/{args.p_epochs}, Loss: {average_loss}: hits@k: {hits_at_k}, mrr: {mrr}")

    torch.save(model.state_dict(), get_model_path(args)) # .replace(".pt", "_blcok.pt")
    print("prompt model saved.")


def match_test(args, test_embeds, test_imgs, k_values, true_class_img):
    clip_model = load_clip_to_cpu(args.backbone)
    model_path = get_model_path(args)
    print(f"test model path: {model_path}")

    # classes = list(test_embeds.keys())
    model = MatchCustomCLIP(args, clip_model).to(args.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dataset = MatchCustomDataset(test_imgs, clip_model.visual)
    dataloader = DataLoader(dataset, batch_size=args.p_batch_size, shuffle=False)
    print(f"Number of test batches: {len(dataloader)}")

    classnames = list(test_embeds.keys())
    embs = list(test_embeds.values())
    model.prompt_learner.initialize_prompts(classnames)

    pred_class_img = {}
    similarity = []
    with torch.no_grad():
        for batch_img, batch_path in dataloader:
            batch_img = batch_img.to(args.device)
            
            logits = model(batch_img, args.device, classnames, embs, args.task) # image-text
            # print(f"logits: {logits}")
            similarity.append(logits)
        
        similarity = torch.cat(similarity, dim=1)
        print(f"simi: {similarity.shape}")

        if args.task == "kgcom":
            _, predicted = torch.topk(similarity, args.tops, dim=1) # retriveal
            _, pred_class_img = retrieve_images(predicted, test_imgs, classnames, pred_class_img)
            
        else:
            _, predicted = torch.topk(similarity, 1, dim=1) # classification
            _, pred_class_img = image_classification(predicted, test_imgs, classnames, pred_class_img)
        
        print(f"pred_class_img: {len(pred_class_img)}")

            # _, predicted = torch.topk(logits, args.tops, dim=1)  # (batch image, tops)
            # for i in range(len(batch_path)):
            #     top_k_txts = [classnames[index] for index in predicted[i]]
            #     for cl in top_k_txts:
            #         if cl not in pred_class_img.keys():
            #             pred_class_img[cl] = []
            #         pred_class_img[cl].append(batch_path[i])

        hits_at_k, mrr = calculate_hits_mrr(true_class_img, pred_class_img, k_values)
    
    print(f"Test hits@k: {hits_at_k}, mrr: {mrr}")

    
def prompt_pipline(args, train_img, train_embs, 
                   test_unseen_classes, test_unseen_img, test_unseen_embs, k_values, label_freq, label_neighbors):
    
    if args.p_mode =="train":
        # train(args, train_classes, train_embs, train_img)
        true_class_img = true_targets(train_img, args.data)
        match_train(args, train_embs, train_img, k_values, true_class_img, label_freq, label_neighbors)

    # test(args, test_seen_classes, test_unseen_classes, test_seen_embs, test_unseen_embs, test_unseen_img, test_seen_img)

    test_classes = test_unseen_classes
    test_img = test_unseen_img
    test_emb = {**test_unseen_embs}
    true_class_img = true_targets(test_img, args.data)

    match_test(args, test_emb, test_img, k_values, true_class_img)
    


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, image_data, text_data):
        self.image_data = image_data
        self.text_data = text_data
        self.transform = Compose([Resize((224, 224), interpolation=Image.BICUBIC), CenterCrop(224), 
                                  lambda image: image.convert("RGB"), ToTensor(), 
                                  Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path)
        image = self.transform(image)

        text = self.text_data[index]

        return image, img_path, text

    def __len__(self):
        return len(self.image_data)
    
def get_partitions(simi_matrix, num_part, images, classes):
    partitions = min_cut_partition(simi_matrix, num_part)
    datasets = []
    for partition in partitions:
        image_partition = [images[i] for i in partition]
        text_partition = [classes[i] for i in partition]
        dataset = PairDataset(image_partition, text_partition)
        datasets.append(dataset)

    return datasets

   

# partitions = get_partitions(simi_matrix, num_part, train_imgs, train_classes)


def pair_match_train(args, train_embs, k_values, true_class_img, partitions):
    clip_model = load_clip_to_cpu(args.backbone)
    torch.manual_seed(args.seed)
    
    # train_class = list(train_embeds.keys())
    model = MatchCustomCLIP(args, clip_model)
    model = model.to(args.device)
    # print(f"model: {model}")

    print("Turning off gradients in both the image and the text encoder")
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    # model.logit_scale.requires_grad = False


    # Double check
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")

    optimizer = optim.Adam(model.prompt_learner.parameters(), lr=args.p_lr) # Only optimize PromptLearner parameters
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

    dataloaders = [DataLoader(dataset, batch_size=args.p_batch_size, shuffle=True) for dataset in partitions]
    print(f"Number of train dataloaders: {len(dataloaders)}")


    for epoch in range(args.p_epochs):
        # if epoch > 1: break

        total_loss = 0
        batch = 0
        epoch_pred = {}

        parti_logits = torch.tensor([]).to(args.device)
        parti_paths = []

        for i, dataloader in enumerate(dataloaders): # 在第i个partition中处理images和texts
            for batch in dataloader:
                images, img_paths, classnames = batch
                embs = [train_embs[cl] for cl in classnames]

                model.prompt_learner.initialize_prompts(classnames)
                optimizer.zero_grad()
                images = images.to(args.device)

                for param in model.parameters():
                    param.to(args.device)

                logits = model(images, args.device, classnames, embs, args.task) # (batch image, block size)
                parti_logits = torch.cat((parti_logits, logits), dim=1)

            _, predicted = torch.topk(parti_logits, 1, dim=1) # pos: top 1
            parti_paths.append(img_paths)

            if args.task == "kgcom":
                pred_class_img, epoch_pred = retrieve_images(predicted, parti_paths, classnames, epoch_pred)
            else:
                pred_class_img, epoch_pred = image_classification(predicted, parti_paths, classnames, epoch_pred)

    
            parti_loss = constractive_loss(parti_logits) 
            parti_loss.backward(retain_graph=True)
            optimizer.step()
        
            hits_at_k, mrr = calculate_hits_mrr(true_class_img, pred_class_img, k_values)
            print(f"{epoch+1}/{args.p_epochs} partition {i+1}/{len(dataloaders)} loss: {parti_loss}, hits@k: {hits_at_k}, mrr: {mrr}")
        
            batch += 1
            total_loss += parti_loss

        hits_at_k, mrr = calculate_hits_mrr(true_class_img, epoch_pred, k_values)
        average_loss = total_loss / len(dataloader)
        print(f"Epoch: {datetime.datetime.now()} - {epoch+1}/{args.p_epochs}, Loss: {average_loss}: hits@k: {hits_at_k}, mrr: {mrr}")

    torch.save(model.state_dict(), get_model_path(args)) # .replace(".pt", "_blcok.pt")
    print("prompt model saved.")