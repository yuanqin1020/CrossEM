
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from argparse import ArgumentParser
import requests
from PIL import Image
import os
import copy
import json

from transformers import CLIPProcessor, CLIPModel

from lavis.models import load_model_and_preprocess

import argparse

from loader import *
from graph_model import *
from prompt import *
from utils import *
from baseline import patches_method, patches_simi_cache


"""
    https://github.com/salesforce/LAVIS
    Image Captioning
"""
class ConditionedText:
    def __init__(self, device):
        self.device = device
        
        self.model, self.processor, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl")

        print("blip flan t5 model load finish.")
                
        self.model.eval()
        self.model.to(self.device)
        
        clip_model_name = ".../openai/clip-vit-base-patch32"
        self.clip_model = CLIPModel.from_pretrained(clip_model_name, local_files_only=True).to(device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name, local_files_only=True)
        
        
    def score_captions(self, image, captions):
        inputs = self.clip_processor(text=captions, images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.clip_model(**inputs)

        batch_size, num_captions = 1, len(captions)
        text_embeds = outputs["text_embeds"].reshape(batch_size, num_captions, -1)
        image_embeds = torch.repeat_interleave(outputs["image_embeds"].unsqueeze(1), num_captions, 1)

        similarity_logits = torch.einsum('bij,bij->bi', text_embeds, image_embeds)[0].cpu().numpy() * 100
        similarity_logits = similarity_logits.round(2)
        order = np.argsort(similarity_logits)
        ordered_captions = [captions[i] for i in order[::-1]]
        scores = list(similarity_logits[order[::-1]].astype("float64").round(2))
        return ordered_captions, scores

    
    def image_caption(self, image_src, num_captions=10):
        
        image = Image.open(image_src).convert("RGB")
        batch = self.processor["eval"](image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            captions = self.model.generate(
                {"image": batch}, 
                use_nucleus_sampling=True, 
                num_captions=num_captions,  
                top_p=0.95,  
                temperature=1.0, 
                num_beams=1  
            )
            
        ordered_captions, scores = self.score_captions(image, captions)
        return ordered_captions, scores
    

class GuidanceModel:
    def __init__(self, device):
        self.device = device
        self.init_models()
        self.ref_image = None
    
    def init_models(self):     
        print("Initializing models .. ")
        self.image_caption_model = ConditionedText(self.device)
        
        print("Guidence model initialization finished!")

    
    def image_to_guidance(self, image_path):    
        with torch.no_grad():
            caption, scores = self.image_caption_model.image_caption(image_path)
            
        return caption, scores

def gen_img_json(file, img_paths):
    json_list = [{"image_path": item} for item in img_paths]
    file = file + ".json"

    if os.path.exists(file):
        return file

    with open(file, "w") as outfile:
        for json_obj in json_list:
            json.dump(json_obj, outfile)
            outfile.write('\n')

    print(f"output img json complete: {file}")
    return file


def gen_guidence(dir, task, img_paths, mode, device):
    json_file = gen_img_json(dir + mode + "_" + task, img_paths)

    model = GuidanceModel(device)
    data = [json.loads(line) for line in open(json_file).readlines()]
    
    new_data = copy.deepcopy(data)
    
    all_records = []
    for k, instance in tqdm(enumerate(data)):
        output = model.image_to_guidance(instance['image_path'])
        all_records.append(output)
            
        new_data[k]["caption"] = output[0]
        new_data[k]["scores"] = output[1]
        
    file = json_file.replace(".json", "_with_guidance.json")
    with open(file, "w") as f:
        for line in new_data:
            f.write(json.dumps(line) + "\n")    

    print(f"guidence complete, write into {file}")


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="...", help="path to dataset") 
    parser.add_argument('--bert_name', default='bert-base-uncased', type=str, help="Pretrained language model name, bart-base or bart-large")
    parser.add_argument("--clip", type=str, default="", help="")
    parser.add_argument("--output", type=str, default="output/", help="output directory")
    parser.add_argument("--cache", type=str, default="cache/", help="")
    parser.add_argument('--device', default='cuda:0', type=str, help="cuda or cpu")

    """ task parameters """
    parser.add_argument("--data", type=str, default="datasets/FB15k/", help={"FB15k", "WN18", "CUB_200", "AWA2", "SUN"})
    # parser.add_argument("--split", type=str, default="split/", help="")
    parser.add_argument("--image", type=str, default="images/", help="")
    parser.add_argument('--train_class_loc', default='split/trainvalclasses.txt', type=str, help='classes to train')
    parser.add_argument('--test_unseen_loc', default='split/testclasses.txt', type=str, help='classes to test unseen')
    parser.add_argument('--test_seen_loc', default='split/valclasses1.txt', type=str, help='classes to test seen')


    """ generation parameters """
    parser.add_argument('--gen', type=float, default=False, help="wherther generate guidence")
    parser.add_argument('--patch', type=float, default=True, help="patch similarity")
    parser.add_argument('--new_trip', type=str, default="", help="new")

    args = parser.parse_args()

    print(f"args: {args}")

    dataset = args.data.strip().split("/")[-2]

    if dataset in ["CUB_200", "AWA2", "SUN"]:
        train_classes = read_classes(args.data, args.root + args.data + args.train_class_loc)
        test_unseen_classes = read_classes(args.data, args.root + args.data + args.test_unseen_loc)
        test_seen_classes = read_classes(args.data, args.root + args.data + args.test_seen_loc)
        train_img, test_unseen_img, test_seen_img = read_image(args.root + args.data + args.image, 
                                                                train_classes, test_unseen_classes, test_seen_classes)

        if args.gen: 
            task = args.data.strip().split("/")[-2] # data = "datasets/img_cf/CUB_200/"
            gen_guidence(args.root + "cache/guidance/", task, train_img, "train", args.device)
            gen_guidence(args.root + "cache/guidance/", task, test_unseen_img, "test_unseen", args.device)
            gen_guidence(args.root + "cache/guidance/", task, test_seen_img, "test_seen", args.device)
            print(f"generate complete for {args.data}")

        if args.patch:
            handle_triples(args.root + args.data + "triplets.txt", args.root + args.data + "new_triplets.txt")
            file = 'new_triplets.txt' if args.new_trip != "" else 'triplets.txt'
            patches_method(args, train_img, args.device, dataset, "train", args.new_trip, file)
            print(f"Patching complete for {dataset}")

    else:
        images = read_image_path(args.root + args.data + args.image)
        ratio = int(0.7 * len(images))
        train_img = images[:ratio]
        print(f"Number of train images: {len(train_img)}")
        patches_simi_cache(args, train_img, args.device, dataset, "train", args.new_trip, "train_triplets.txt")
