import argparse
import multiprocessing
import torch
import re


def _get_device_count():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return -1


def parse_arguments():
    parser = argparse.ArgumentParser(description="Sequence Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # reproducibility
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--deterministic', action='store_true', default=False)
    
    # dataset
    parser.add_argument("--cached_train_dataset", type=str, help="Path of cached pickle train dataset")
    parser.add_argument("--city", type=str, default='', help='subset of cities from train set')
    parser.add_argument("--seq_length", type=int, default=15,
                        help="Number of images in each sequence")
    parser.add_argument("--reverse", action='store_true', default=False, help='reverse DB sequences frames')
    parser.add_argument("--cut_last_frame", action='store_true', default=False, help='cut last sequence frame')
    parser.add_argument("--val_posDistThr", type=int, default=25, help="_")
    parser.add_argument("--train_posDistThr", type=int, default=10, help="_")
    parser.add_argument("--negDistThr", type=int, default=25, help="_")
    parser.add_argument('--img_shape', type=int, default=[480, 640], nargs=2,
                        help="Resizing shape for images (HxW).")

    # about triplets and mining
    parser.add_argument("--nNeg", type=int, default=10,
                        help="How many negatives to consider per each query in the loss")
    parser.add_argument("--cached_negatives", type=int, default=1000,
                        help="How many negatives to use to compute the hardest ones")
    parser.add_argument("--cached_queries", type=int, default=1000,
                        help="How many queries to keep cached")
    parser.add_argument("--queries_per_epoch", type=int, default=5000,
                        help="How many queries to consider for one epoch. Must be multiple of cached_queries")

    # models
    parser.add_argument("--arch", type=str, default="r18l3",
                        choices=["vgg16", "r18l3", "r18l4", "r50l3", "r50l4", "r101l3", "r101l4", "vit",
                                 "cct224", "cct384", 'timesformer'],
                        help="_")
    parser.add_argument("--pooling", type=str, default="none", choices=["netvlad", "gem", "spoc", "mac", "rmac",
                                                                       "none",  "_"])
    parser.add_argument("--aggregation", type=str, default="seqvlad",
                        choices=["cat", "fc", "seqvlad", "_"])
    parser.add_argument("--freeze_layer", type=str, default="layer3",  
                        choices=["layer1", "layer2", "layer3", "layer4"], help="_")
    parser.add_argument("--trunc_te", type=int, default=None, choices=list(range(0, 14)))
    parser.add_argument("--freeze_te", type=int, default=None, choices=list(range(-1, 14)))
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    parser.add_argument("--pretrain_model", type=str, default=None,
                        help="Path to load pretrained model from.")

    parser.add_argument("--fc_out", type=int, help='output size if aggregation=fc', default=768)
    parser.add_argument("--pca_outdim", type=int, help='output size with PCA', default=None)

    # training pars
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--n_gpus", type=int, default=_get_device_count())
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count(),
                        help="num_workers for all dataloaders")

    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Number of triplets: (query + pos + negs) * seq_length.")
    parser.add_argument("--infer_batch_size", type=int, default=8,
                        help="Batch size for inference (caching and testing)")

    parser.add_argument("--epochs_num", type=int, default=100,
                        help="number of epochs to train for")
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--lr", type=float, default=0.00001, help="_")
    parser.add_argument("--lr_aggregator", type=float, default=0.0001, help="_")
    parser.add_argument('--multiple_lr', action='store_true', default=False)
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay value")
    parser.add_argument("--optim", type=str, default="adam", help="_", choices=["adam", "sgd"])
    parser.add_argument("--criterion", type=str, default='triplet', help='loss to be used',
                        choices=["triplet", "sare_ind", "sare_joint"])
    parser.add_argument("--margin", type=float, default=0.1,
                        help="margin for the triplet loss")
        
    # PATHS
    parser.add_argument("--dataset_path", type=str, default='', help="Path of the dataset")
    parser.add_argument("--exp_name", type=str, default="default",
                        help="Folder name of the current run (saved in ./runs/)")


    args = parser.parse_args()
        
    if args.queries_per_epoch % args.cached_queries != 0:
        raise ValueError("Please ensure that queries_per_epoch is divisible by cache_refresh_rate, " +
                         f"because {args.queries_per_epoch} is not divisible by {args.cached_queries}")
    
    return args

