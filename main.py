import argparse
import torch
import math

from loader import *
from graph_model import *
from prompt import *
from match import *
from utils import *
from guidance import *
from baseline import *
from generate import gen_guidence


os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "0,1"

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/home/zhangjunyu/yq/Align/", help="path to dataset") #   /home/yuanqin/Align/
    parser.add_argument('--bert_name', default='bert-base-uncased', type=str, help="Pretrained language model name, bart-base or bart-large")
    parser.add_argument("--clip", type=str, default="", help="")
    parser.add_argument("--output", type=str, default="output/", help="output directory")
    parser.add_argument("--cache", type=str, default="cache/", help="")
    parser.add_argument('--device', default='cuda:1', type=str, help="cuda or cpu")

    """ task parameters """
    parser.add_argument("--data", type=str, default="datasets/SUN/", help={"FB15k", "WN18", "CUB_200", "AWA2", "SUN"})
    parser.add_argument("--image", type=str, default="images/", help="")
    # kg completion
    parser.add_argument('--kg', default='kg/', type=str, help='kg data')


    # image classification
    parser.add_argument('--train_class_loc', default='split/trainvalclasses.txt', type=str, help='classes to train')
    parser.add_argument('--test_unseen_loc', default='split/testclasses.txt', type=str, help='classes to test unseen')
    parser.add_argument('--test_seen_loc', default='split/valclasses1.txt', type=str, help='classes to test seen')
    

    """ graph training parameters """
    parser.add_argument('--mode', type=str, default="test")
    parser.add_argument('--num_epochs', default=30, type=int, help="Training epochs")
    parser.add_argument('--batch_size', default=528, type=int, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument('--g_type', type=str, default="SAGE", help="GNN, GAT, GCN, SAGE")
    parser.add_argument("--seed", type=int, default=7, help="only positive value enables a fixed seed")
    parser.add_argument("--hidden_channels", type=int, default=256) 
    parser.add_argument("--out_channels", type=int, default=128)
    parser.add_argument('--agg_fun', type=str, default="MEAN", help="MEAN, MAX")


    """ prompt parameters """
    parser.add_argument('--p_mode', type=str, default="train")
    parser.add_argument('--p_metric', type=str, default="gzsl", help="gzsl, zsl")
    parser.add_argument('--backbone', type=str, default="ViT-B/16")
    parser.add_argument('--n_ctx', default=4, type=int)
    parser.add_argument('--ctx_init', type=str, default="init", help={"random", "init"})  
    parser.add_argument('--ctx_bais', type=str, default="unbais", help={"unbais", "cluster", "meta"}) 
    parser.add_argument('--clss', type=str, default="SAGE", help={"freq, feat, SAGE, GNN"})
    parser.add_argument('--p_batch_size', default=128, type=int, help="batch size") 
    parser.add_argument("--tk_w", type=float, default=0.2)  
    parser.add_argument("--p_lr", type=float, default=0.00001)
    parser.add_argument('--p_epochs', type=int, default=2)
    parser.add_argument("--p_clusters", type=int, default=5) 
    parser.add_argument('--tops', type=int, default=10)

    parser.add_argument('--c_prune', type=float, default=0.012, help={"whether to prune irrelated pairs"})
    parser.add_argument('--theta_1', type=float, default=0.0, help={"variance"})
    parser.add_argument('--theta_2', type=float, default=0.0, help={"mean"})
    parser.add_argument('--pcp', type=bool, default=True)
    parser.add_argument('--hns', type=bool, default=True)
    parser.add_argument('--oloss', type=bool, default=True)
    parser.add_argument('--scala', type=str, default=".6", help=".2, .3, .6")
    


    """ optimization parameters """
    parser.add_argument('--block_rate', type=int, default=100)
    parser.add_argument('--blocker', type=str, default="cluster", help={"random", "cluster"})
    parser.add_argument("--cluster_num", type=int, default=10) 
    parser.add_argument('--bl', type=str, default="pcp", help={"pcp", "hash", "simi"})

    parser.add_argument("--fusion", default="concat-image", help="How to fuse the additional guidance.")
    parser.add_argument("--combine", default="logits", help="logits, features, guidance")
    parser.add_argument('--new_trip', type=str, default="", help="new")

    parser.add_argument('--sample_num', type=int, default=20)
    parser.add_argument('--method', type=str, default="guided", help="base, hard, sentence, prompt, guided")


    
    args = parser.parse_args()

    return args


    
if __name__ == "__main__":
    args = setup_args()
    print(f"args: {args}")

    k_values = [1, 3, 5, 10]

    dataset = args.data.strip().split("/")[-2]

    if dataset in ["CUB_200", "AWA2", "SUN"]:
        train_classes = read_classes(args.data, args.root + args.data + args.train_class_loc)
        test_unseen_classes = read_classes(args.data, args.root + args.data + args.test_unseen_loc)
        test_seen_classes = read_classes(args.data, args.root + args.data + args.test_seen_loc)
        train_img, test_unseen_img, test_seen_img = read_image(args.root + args.data + args.image, 
                                                            train_classes, test_unseen_classes, test_seen_classes)

        print(f"Number of train classes: {len(train_classes)}, images: {len(train_img)}")
        print(f"Number of test unseen classes: {len(test_unseen_classes)}, images: {len(test_unseen_img)}")
        print(f"Number of test seen classes: {len(test_seen_classes)}, images: {len(test_seen_img)}")

        if args.method == "base":
            base_method(args, test_unseen_classes, test_unseen_img)
        elif args.method == "hard":
            hard_prompt_method(args, test_unseen_classes, test_unseen_img, dataset)
        elif args.method == "sentence":
            sentence_method(args, test_unseen_classes, dataset, test_unseen_img)
    
        else:
            _, train_embs, test_seen_embs, test_unseen_embs, train_data, data, label_freq, label_neighbors = graph_processor(
                args, train_classes, test_seen_classes, test_unseen_classes, dataset)
            
            suffix = "_new.txt" if args.new_trip != "" else ".txt"
            patch_file = args.root + "cache/feat/logits_preprocess_" + dataset + "_train" + suffix
            classes_images_simi = preprocess_classes_image(train_data, train_classes, patch_file, args.root, dataset)
            
            if args.blocker != "":
                block_num = math.ceil(len(train_embs) / args.block_rate)
                train_sets = block_graph(train_embs, block_num, args.blocker)
            else:
                train_sets = [train_embs]
            print(f"Number of datasets by blocking: {len(train_sets)}")


            if args.method == "prompt":
                prompt_pipline(args, train_img, train_sets, test_unseen_classes, test_unseen_img, test_unseen_embs, 
                               k_values, label_freq, label_neighbors)

            elif args.method == "guided":
                # _, train_embs, test_seen_embs, test_unseen_embs, train_data = graph_processor(args, train_classes, test_seen_classes, test_unseen_classes)
                guided_pipline(args, train_img, train_sets, test_unseen_img, test_unseen_embs, dataset, 
                               k_values, classes_images_simi, train_data, data, label_freq, label_neighbors)

    else:
        entity2text, relation2text = get_ent_rel_mapping(args.root + args.data + args.kg, args.scala)
        handle_kg_data(args.root + args.data + args.kg, entity2text, relation2text)
        
        train_embs, train_data, train_label_neighbors, train_label_freq = kg_graph_train(args)

        images = read_image_path(args.root + args.data + args.image, dataset, entity2text, train_data.y)
        # true_class_img = true_targets(images, args.data, entity2text)
        print(f"Number of total images: {len(images)}")

        ratio = int(0.7 * len(images))
        train_img = images[:ratio]
        test_img = images[ratio:]
        # patches_simi_cache(args, train_img, args.device, dataset, "train", args.new_trip, "train_triplets.txt")
        # gen_guidence(args.root + "cache/guidance/", dataset, test_img, "train", args.device)

        test_embs, test_data, test_label_neighbors, test_label_freq = kg_graph_infer(args)
        print(f"Number of train entities and relations: {len(train_data.x)}, {train_data.edge_index.shape}, train images: {len(train_img)}")
        print(f"Number of test entities and relations: {len(test_data.x)}, {test_data.edge_index.shape}, images: {len(test_img)}")

        # train_classes = remove_leaf(train_embs, train_data)
        # test_classes = remove_leaf(test_embs, test_data)

        train_classes = list(train_embs.keys())
        test_classes = list(test_embs.keys())


        if args.method == "base":
            base_method(args, test_classes, test_img, entity2text)
        elif args.method == "hard":
            hard_prompt_method(args, test_classes, test_img, dataset, entity2text)
        elif args.method == "sentence":
            sentence_method(args, test_classes, dataset, test_img)
    
        else:
            patch_file = args.root + "cache/feat/logits_preprocess_" + dataset + "_train_1" + args.scala + ".txt"
            classes_images_simi = preprocess_classes_image(train_data, train_classes, patch_file, args.root, dataset)
            
            if args.blocker != "":
                block_num = math.ceil(len(train_embs) / args.block_rate)
                train_sets = block_graph(train_embs, block_num, args.blocker)
            else:
                train_sets = [train_embs]
            print(f"Number of datasets by blocking: {len(train_sets)}")

            if args.method == "prompt":
                prompt_pipline(args, train_img, train_sets, test_unseen_classes, test_unseen_img, test_unseen_embs, 
                               k_values, train_label_neighbors, train_label_freq)

            elif args.method == "guided":
                if args.p_mode =="train":
                    true_class_img = true_targets(train_img, args.data, entity2text)
                    
                    guided_train(args, train_sets, train_img, k_values, true_class_img, 
                                classes_images_simi, train_data, train_label_freq, train_label_neighbors)

                true_class_img = true_targets(test_img, args.data, entity2text)
                guided_test(args, test_img, test_embs, k_values, true_class_img, test_data, test_label_freq, test_label_neighbors)



