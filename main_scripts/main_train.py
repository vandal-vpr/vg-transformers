from init_scripts import init_script

init_script()
import math
import torch
from tqdm import tqdm
import logging
import random
import numpy as np
import torch.nn as nn
from os.path import join
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
import re

from tvg.models import TVGNet
from tvg.datasets import BaseDataset, TrainDataset, collate_fn, retrieve_db_object
from tvg.evals import test
from tvg.utils import (parse_arguments, setup_logging, save_checkpoint, resume_train,
                       load_pretrained_backbone, configure_transform)


#### Initial setup: parser, logging...
args = parse_arguments()
start_time = datetime.now()

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

torch.backends.cudnn.benchmark = True
if args.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args.output_folder = join("logs", args.exp_name, str(args.seed))


setup_logging(args.output_folder)

logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")

#### Creation of Datasets
logging.debug(f"Loading datasets from directory {args.dataset_path}")

# get transform
meta = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
img_shape = (args.img_shape[0], args.img_shape[1])
transform = configure_transform(image_dim=img_shape, meta=meta)

if args.cached_train_dataset is not None:
    logging.info(f"Retrieving train set from cached: {args.cached_train_dataset}")
    triplets_ds = retrieve_db_object(args.cached_train_dataset)
    triplets_ds.bs = args.infer_batch_size
    triplets_ds.seq_len = args.seq_length
    triplets_ds.n_gpus = args.n_gpus
    triplets_ds.transform = transform
    triplets_ds.img_shape = args.img_shape
    triplets_ds.base_transform = transform
    triplets_ds.nNeg = args.nNeg
    triplets_ds.cut_last_frame = args.cut_last_frame
    if args.cut_last_frame:
        for i, seq in enumerate(triplets_ds.images_paths):
            triplets_ds.images_paths[i] = ','.join((seq.split(',')[:-1]))
        for i, seq in enumerate(triplets_ds.db_paths):
            triplets_ds.db_paths[i] = ','.join((seq.split(',')[:-1]))
        for i, seq in enumerate(triplets_ds.q_paths):
            triplets_ds.q_paths[i] = ','.join((seq.split(',')[:-1]))
else:
    logging.info("Loading train set...")
    triplets_ds = TrainDataset(cities=args.city, dataset_folder=args.dataset_path, split='train',
                               base_transform=transform, seq_len=args.seq_length, cut_last_frame=args.cut_last_frame,
                               pos_thresh=args.train_posDistThr,
                               neg_thresh=args.negDistThr, infer_batch_size=args.infer_batch_size,
                               n_gpus=args.n_gpus, img_shape=args.img_shape,
                               cached_negatives=args.cached_negatives,
                               cached_queries=args.cached_queries, nNeg=args.nNeg)

logging.info(f"Train set: {triplets_ds}")
logging.info("Loading val set...")
val_ds = BaseDataset(dataset_folder=args.dataset_path, split='val',
                     base_transform=transform, seq_len=args.seq_length, cut_last_frame=args.cut_last_frame,
                     pos_thresh=args.val_posDistThr)
logging.info(f"Val set: {val_ds}")

logging.info("Loading test set...")
test_ds = BaseDataset(dataset_folder=args.dataset_path, split='test',
                      base_transform=transform, seq_len=args.seq_length, cut_last_frame=args.cut_last_frame,
                      pos_thresh=args.val_posDistThr)
logging.info(f"Test set: {test_ds}")

#### Initialize model
model = TVGNet(args)
model = model.to(args.device)
if args.pooling in ["netvlad"]:
    if not args.resume and not args.pretrain_model:
        triplets_ds.is_inference = True
        model.pooling.initialize_netvlad_layer(args, triplets_ds, model.encoder)
        triplets_ds.is_inference = False

if args.aggregation in ["seqvlad"]:
    if not args.resume:
        triplets_ds.is_inference = True
        model.aggregator.initialize_seqvlad_layer(args, triplets_ds, model.encoder)
        triplets_ds.is_inference = False

model = torch.nn.DataParallel(model)
triplets_ds.features_dim = args.features_dim
logging.info(f"Output dimension of the model is {model.module.meta['outputdim']}")
#### Setup Optimizer and Loss
if args.optim == "adam":
    if args.multiple_lr:
        optimizer = torch.optim.Adam([
            {'params': model.module.encoder.parameters()},
            {'params': model.module.pooling.parameters()},
            {'params': model.module.aggregator.parameters(), 'lr': args.lr_aggregator}
        ],
            lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

if args.criterion == "triplet":
    criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")

#### Resume model, optimizer, and other training parameters
if args.resume:
    model, optimizer, best_r5, start_epoch_num, not_improved_num = resume_train(args, model, optimizer)
    logging.info(f"Resuming from epoch {start_epoch_num} with best recall@5 {best_r5:.1f}")
elif args.pretrain_model and args.arch != 'Official-timesf':
    model = load_pretrained_backbone(args, model)
    best_r5 = start_epoch_num = not_improved_num = 0
else:
    best_r5 = start_epoch_num = not_improved_num = 0

#### Training loop
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: [{epoch_num:02d}]")

    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0, 1), dtype=np.float32)

    # How many loops should an epoch last
    loops_num = math.ceil(args.queries_per_epoch / args.cached_queries)
    for loop_num in range(loops_num):
        logging.debug(f"Cache: {loop_num + 1} / {loops_num}")

        # creates triplets on the smaller cache set
        triplets_ds.compute_triplets(model)
        triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                 batch_size=args.train_batch_size,
                                 collate_fn=collate_fn,
                                 pin_memory=(args.device == "cuda"),
                                 drop_last=True)

        model = model.train()
        for images, _, _ in tqdm(triplets_dl, ncols=100):
            # images shape: (bsz, seq_len*(nNeg + 2), 3, H, W)
            # triplets_local_indexes shape: (bsz, nNeg+2) -> contains -1 for query, 1 for pos, 0 for neg

            # reshape images to only have 4-d
            images = images.view(-1, 3, *img_shape)

            # features : (bsz*(nNeg+2), model_output_size)
            features = model(images.to(args.device))

            # Compute loss by passing the triplets one by one
            loss_triplet = 0

            features = features.view(args.train_batch_size, -1, args.features_dim)
            if args.criterion == "triplet":
                for b in range(args.train_batch_size):
                    query = features[b:b + 1, 0]  # size (1, output_dim)
                    pos = features[b:b + 1, 1]  # size (1, output_dim)
                    negatives = features[b, 2:]  # size (nNeg, output_dim)
                    # negatives has 10 images , pos and query 1 but
                    # the loss yields same result as calling it 10 times

                    loss_triplet += criterion_triplet(query,
                                                      pos,
                                                      negatives)
            del features

            loss_triplet /= (args.train_batch_size * args.nNeg)
            optimizer.zero_grad()
            loss_triplet.backward()
            optimizer.step()

            # Keep track of all losses by appending them to epoch_losses
            batch_loss = loss_triplet.item()
            epoch_losses = np.append(epoch_losses, batch_loss)
            del loss_triplet

        logging.debug(f"Epoch[{epoch_num:02d}]({loop_num + 1}/{loops_num}): " +
                      f"current batch triplet loss = {batch_loss:.4f}, " +
                      f"average epoch triplet loss = {epoch_losses.mean():.4f}")

    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average epoch triplet loss = {epoch_losses.mean():.4f}")

    # Compute recalls on validation set
    recalls, recalls_str = test(args, val_ds, model)
    logging.info(f"Recalls on val set: {recalls_str}")

    is_best = recalls[1] > best_r5

    # Save checkpoint, which contains all training parameters
    save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
                           "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls, "best_r5": best_r5,
                           "not_improved_num": not_improved_num
                           }, is_best, filename="last_model.pth")

    # If recall@5 did not improve for "many" epochs, stop training
    if is_best:
        logging.info(f"Improved: previous best R@5 = {best_r5:.1f}, current R@5 = {recalls[1]:.1f}")
        best_r5 = recalls[1]
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(f"Not improved: {not_improved_num} / {args.patience}: best R@5 = {best_r5:.1f}, current R@5 = {recalls[1]:.1f}")
        if not_improved_num >= args.patience:
            logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            break

logging.info(f"Best R@5: {best_r5:.1f}")
logging.info(f"Trained for {epoch_num:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set
best_model_state_dict = torch.load(join(args.output_folder, "best_model.pth"))["model_state_dict"]
model.load_state_dict(best_model_state_dict)

recalls, recalls_str = test(args, test_ds, model)
logging.info(f"Recalls on test set: {recalls_str}")
