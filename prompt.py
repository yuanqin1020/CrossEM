import argparse
import datetime
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from loader import *
from utils import *
from evaluate import *
from graph_model import graph_processor


from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

def get_clusters(ver_embeds, num_clusters):
    ver_reps = torch.stack(ver_embeds)
    ver_arrays = ver_reps.detach().numpy()

    cluster_alg = KMeans(n_clusters=num_clusters, n_init=10)
    clusters = cluster_alg.fit_predict(ver_arrays)

    cluster_context = []
    for cluster_idx in range(num_clusters):
        cluster_vertices = [ver_embeds[i] for i, cluster in enumerate(clusters) if cluster == cluster_idx]
        centroid = torch.mean(torch.stack(cluster_vertices), dim=0)
        cluster_context.append(centroid)

    cluster_context = torch.stack(cluster_context)

    return cluster_context


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.images = image_paths
        self.transform = Compose([Resize((224, 224), interpolation=Image.BICUBIC), CenterCrop(224), 
                                  lambda image: image.convert("RGB"), ToTensor(), 
                                  Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # vertex = self.vertices
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
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
    
class MetaNet(nn.Module):
    def __init__(self, vis_dim, ctx_dim):
        super(MetaNet, self).__init__()
        self.linear1 = nn.Linear(vis_dim, vis_dim // 2)  
        self.relu = nn.LeakyReLU(inplace=True) 
        self.linear2 = nn.Linear(vis_dim // 2, ctx_dim)  

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        return out
    

class PromptLearner(nn.Module):

    def __init__(self, classnames, ver_embeds, num_clusters, clip_model, ctx_init, ctx_bais, n_ctx, clss, cluster_dim, tk_w, device):
        super(PromptLearner, self).__init__()
        self.num_clusters = num_clusters
        self.ver_embeds = ver_embeds
        self.n_cls = len(classnames)
        self.device = device
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.vis_dim = clip_model.visual.output_dim
        self.dtype = clip_model.dtype
        self.n_ctx = n_ctx
        self.ctx_bais = ctx_bais
        self.clss = clss
        

        if ctx_init == "init":
            ctx_init = "a photo of a"
            print("Use given words to initialize context prefix vectors")
            ctx_init = ctx_init.replace("_", " ")
            prompt_prefix = ctx_init
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]

        else:
            print("Use random context prefix initialization")
            prompt_prefix = " ".join(["X"] * self.n_ctx)
            ctx_vectors = torch.empty(n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
        
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {self.n_ctx}")
            
        self.ctx = nn.Parameter(ctx_vectors)
        print(f"self.ctx: {self.ctx}")

        if self.ctx_bais == "cluster":
            self.cluster_reduction = nn.Sequential(nn.Linear(cluster_dim, self.ctx_dim), nn.ReLU())
        elif self.ctx_bais == "meta":
            self.meta_net = nn.Sequential(nn.Linear(self.vis_dim, self.ctx_dim), nn.LeakyReLU())
            self.meta_net.half()
            print(f"self.meta_net: {self.meta_net}")
        
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) 
        with torch.no_grad():
            self.token_embed = clip_model.token_embedding(tokenized_prompts).type(self.dtype) 

        self.token_prefix = self.token_embed[:, :1, :].to(self.device)  
        self.token_suffix = self.token_embed[:, 1 + self.n_ctx :, :].to(self.device)   
        self.tokenized_prompts = tokenized_prompts.to(self.device)  

        if self.clss == "feat":
            self.feat_reduction = nn.Sequential(nn.Linear(len(self.ver_embeds[0]), 77), nn.ReLU()).to(self.device)
            tk_weight_vector = torch.full((77,), tk_w, requires_grad=True)
            self.token_weight = nn.Parameter(tk_weight_vector)

        
    
    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        tokenized_prompts = self.tokenized_prompts
        ctx = self.ctx                   
        ctx = ctx.unsqueeze(0)          

        if self.clss == "feat":
            vers = torch.stack(self.ver_embeds).to(self.device)
            class_emb = self.feat_reduction(vers) # (n_cls, n_tkn)

            token_weight = self.token_weight.unsqueeze(0)
            tokenized_prompts = torch.cat([tokenized_prompts * (1 - token_weight) + class_emb * token_weight], dim=1)

            tokenized_prompts = tokenized_prompts.to(self.device) 

        if self.ctx_bais == "cluster":
            clusters = get_clusters(self.ver_embeds, self.num_clusters).to(self.device)
            projects = self.cluster_reduction(clusters)
            projects = torch.mean(projects, dim=0)      
            bias = projects.unsqueeze(0).unsqueeze(0)       

            ctx_shifted = ctx + bias       
            # print(f"ctx shifted: {ctx_shifted.shape}")

        elif self.ctx_bais == "meta":
            print(f"Meta context bais")
            bias = self.meta_net(im_features) 
            bias = bias.unsqueeze(1)        
            print(f"meta bais: {bias}")

            ctx_shifted = ctx + bias        
            print(f"ctx shifted: {ctx_shifted.shape}")
        
        else:
            ctx_shifted = ctx
        
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  
            prompts.append(pts_i)

        prompts = torch.stack(prompts) 
        
        return prompts, tokenized_prompts
    

    def construct_prompts(self, ctx, prefix, suffix, label=None):

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts.type(self.dtype)
    


class CustomCLIP(nn.Module):
    def __init__(self, classnames, ver_embeds, args, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.ctx_bais = args.ctx_bais

        self.prompt_learner = PromptLearner(classnames, ver_embeds, args.p_clusters, clip_model, args.ctx_init, 
                                            self.ctx_bais, args.n_ctx, args.clss, len(ver_embeds[0]), args.tk_w, args.device)
        

    def forward(self, image, device, label=None):
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts, tokenized_prompts = self.prompt_learner(image_features)

        logits = []
        if self.ctx_bais == "meta":
            for pts_i, imf_i in zip(prompts, image_features):
                text_features = self.text_encoder(pts_i, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                l_i = logit_scale * imf_i @ text_features.t()
                logits.append(l_i)
            logits = torch.stack(logits)

        else:
            for pts_i in prompts:
                text_features = self.text_encoder(pts_i, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits.append(logit_scale * image_features @ text_features.t())

            logits = torch.stack(logits).squeeze()
        
        return logits
    
def train(args, train_classes, ver_embeds, image_paths):
    clip_model = load_clip_to_cpu(args.backbone)
    torch.manual_seed(args.seed)
    
    model = CustomCLIP(train_classes, list(ver_embeds.values()), args, clip_model)
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
    print(f"Parameters to be updated: {enabled}")

    optimizer = optim.Adam(model.prompt_learner.parameters(), lr=args.p_lr)
 
    true_class_img = true_targets(image_paths, args.data)

    dataset = CustomDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=args.p_batch_size, shuffle=True)
    print(f"Number of train batches: {len(dataloader)}")

    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.p_epochs):

        total_loss = 0
        batch = 0
        epoch_pred = {}
        for batch_img, batch_path in dataloader:

            optimizer.zero_grad()
            batch_img = batch_img.to(args.device)

            for param in model.parameters():
                param.to(args.device)

            logits = model(batch_img, args.device) 

            loss = constractive_loss(logits)
            loss.backward(retain_graph=True)
            optimizer.step()

            pred_class_img = {}
            _, predicted = torch.topk(logits, args.tops, dim=1) 
            for i in range(len(batch_path)):
                top_k_txts = [train_classes[index] for index in predicted[i]]
                for cl in top_k_txts:
                    if cl not in pred_class_img.keys():
                        pred_class_img[cl] = []
                    pred_class_img[cl].append(batch_path[i])
                    
                    if cl not in epoch_pred.keys():
                        epoch_pred[cl] = []
                    epoch_pred[cl].append(batch_path[i])

            _, acc, _ = calculate_accuracy(pred_class_img, true_class_img)
            print(f"{epoch+1}/{args.p_epochs} batch {batch+1}/{len(dataloader)} loss: {loss.item()}, acc: {acc}, lr: {args.p_lr}")
            
            batch += 1
            total_loss += loss.item()

        _, accuracy, bug_classes = calculate_accuracy(epoch_pred, true_class_img)

        if epoch == 0:
            print(f"train bug class number: {len(bug_classes)} - {bug_classes}")

        average_loss = total_loss / len(dataloader)
        print(f"Epoch: {datetime.datetime.now()} - {epoch+1}/{args.p_epochs}, Loss: {average_loss}: Acc: {accuracy}")


    torch.save(model.state_dict(), get_model_path(args))
    print("prompt model saved.")



def get_model_path(args):
    path = 'model_' + args.task + '_' + args.g_type + '_' + args.ctx_init + '_' + args.ctx_bais + '_' + args.clss + '_' + args.data.split("/")[-2] + '.pt'
    
    return args.root + args.cache + path

def test(args, test_seen_classes, test_unseen_classes, test_seen_embs, test_unseen_embs, test_unseen_img, test_seen_img):
    clip_model = load_clip_to_cpu(args.backbone)
    model_path = get_model_path(args)
    print(f"test model path: {model_path}")
    
    test_classes = test_unseen_classes + test_seen_classes
    test_img = test_unseen_img + test_seen_img
    test_emb = {**test_unseen_embs, **test_seen_embs}
    acc_avg_mat = test_zsl(test_classes, test_emb, test_img, args, clip_model, model_path)
    print(f"Number of matching test classes: {len(test_classes)}, imges: {len(test_img)}")

    print(f"Matching test avg accuracy: {acc_avg_mat}")


def test_zsl(classes, embs, imgs, args, clip_model, model_path):
    
    model = CustomCLIP(classes, list(embs.values()), args, clip_model).to(args.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    true_class_img = true_targets(imgs, args.data)
    dataset = CustomDataset( imgs)
    dataloader = DataLoader(dataset, batch_size=args.p_batch_size, shuffle=False)
    print(f"Number of test batches: {len(dataloader)}")

    with torch.no_grad():
        pred_class_img = {}
        for batch_img, batch_path in dataloader:
            batch_img = batch_img.to(args.device)
            logits = model(batch_img, args.device) # image-text
            # print(f"logits: {logits}")

            _, predicted = torch.topk(logits, args.tops, dim=1)  # (batch image, tops)
            for i in range(len(batch_path)):
                top_k_txts = [classes[index] for index in predicted[i]]
                for cl in top_k_txts:
                    if cl not in pred_class_img.keys():
                        pred_class_img[cl] = []
                    pred_class_img[cl].append(batch_path[i])

        _, accuracy, bug_classes = calculate_accuracy(pred_class_img, true_class_img)

        print(f"test zsl bug class number: {len(bug_classes)} - {bug_classes}")
        

    return accuracy


def test_gzsl(test_classes, classes, embs, imgs, args, clip_model, model_path):
    
    model = CustomCLIP(classes, list(embs.values()), args, clip_model).to(args.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    true_class_img = true_targets(imgs, args.data)
    dataset = CustomDataset(embs, imgs)
    dataloader = DataLoader(dataset, batch_size=args.p_batch_size, shuffle=False)
    print(f"Number of test batches: {len(dataloader)}")

    with torch.no_grad():
        pred_class_img = {}
        for batch_img, batch_path in dataloader:
            batch_img = batch_img.to(args.device)
            logits = model(batch_img, args.device) # image-text
            # print(f"logits: {logits}")

            _, predicted = torch.topk(logits, args.tops, dim=1)  # (batch image, tops)
            for i in range(len(batch_path)):
                top_k_txts = [classes[index] for index in predicted[i]]
                for cl in top_k_txts:
                    if cl not in test_classes: continue
                    if cl not in pred_class_img.keys():
                        pred_class_img[cl] = []
                    pred_class_img[cl].append(batch_path[i])

        _, accuracy, bug_classes = calculate_accuracy(pred_class_img, true_class_img)

        print(f"test gzsl bug class number: {len(bug_classes)} - {bug_classes}")

    return accuracy

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_class_loc', default='split/trainvalclasses.txt', type=str, help='classes to train')
    parser.add_argument('--test_unseen_loc', default='split/testclasses.txt', type=str, help='classes to test unseen')
    parser.add_argument('--test_seen_loc', default='split/valclasses1.txt', type=str, help='classes to test seen')
    args = parser.parse_args()
    print(f"args: {args}")

    if args.task == "imgcf":
        train_classes = read_classes(args.data, args.root + args.data + args.train_class_loc)
        test_unseen_classes = read_classes(args.data, args.root + args.data + args.test_unseen_loc)
        test_seen_classes = read_classes(args.data, args.root + args.data + args.test_seen_loc)

        _, train_embs, test_seen_embs, test_unseen_embs, train_data = graph_processor(args, train_classes, test_seen_classes, test_unseen_classes)

        train_img, test_unseen_img, test_seen_img = read_image_path(args.root + args.data + args.image, 
                                                                    train_classes, test_unseen_classes, test_seen_classes)

        print(f"Number of train classes: {len(train_classes)}, images: {len(train_img)}, ver embs: {len(train_embs)}")
        print(f"Number of test unseen classes: {len(test_unseen_classes)}, images: {len(test_unseen_img)}, ver embs: {len(test_unseen_embs)}")
        print(f"Number of test seen classes: {len(test_seen_classes)}, images: {len(test_seen_img)}, ver embs: {len(test_seen_embs)}")


        if args.p_mode =="train":
            train(args, train_classes, train_embs, train_img)
        else:
            test(args, test_seen_classes, test_unseen_classes, test_seen_embs, test_unseen_embs, test_unseen_img, test_seen_img)
