import torch
import os

from ..datasets import TrainDataset
from ..utils import configure_transform


def cache_db_object(db, path):
    torch.save(db, path)


def retrieve_db_object(path):
    return torch.load(path)


def cache_mapillary_train(root_dir, cache_path, split='train',
                          posDistThr=10,
                          negDistThr=25,
                          infer_batch_size=8,
                          n_gpus=1,
                          features_dim=256,
                          nNeg=10,
                          cached_queries=1000,
                          cached_negatives=1000,
                          cities='',
                          seq_length=3,
                          cut_last_frame=False):
    # get transform
    meta = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    transform = configure_transform(image_dim=(480, 640), meta=meta)

    train_dataset = TrainDataset(cities=cities, dataset_folder=root_dir, split=split,
                              base_transform=transform,
                              seq_len=seq_length, pos_thresh=posDistThr,
                              neg_thresh=negDistThr, infer_batch_size=infer_batch_size,
                              n_gpus=n_gpus, features_dim=features_dim,
                              cached_negatives=cached_negatives,
                              cached_queries=cached_queries, nNeg=nNeg, cut_last_frame=cut_last_frame)

    cache_db_object(train_dataset, cache_path)
