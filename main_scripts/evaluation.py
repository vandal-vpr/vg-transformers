from collections import OrderedDict
import logging
from datetime import datetime
import einops
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import random
# our imports
from init_scripts import init_script
init_script()
from tvg.datasets import BaseDataset, PCADataset
from tvg.evals import test
from tvg.models import TVGNet
from tvg.utils import parse_arguments, configure_transform, setup_logging


def compute_pca(args, model, transform, full_features_dim):
    model = model.eval()
    pca_ds = PCADataset(dataset_folder=args.dataset_path, split='train',
                        base_transform=transform, seq_len=args.seq_length)
    logging.info(f'PCA dataset: {pca_ds}')
    num_images = min(len(pca_ds), 2 ** 14)
    if num_images < len(pca_ds):
        idxs = random.sample(range(0, len(pca_ds)), k=num_images)
    else:
        idxs = list(range(len(pca_ds)))
    subset_ds = Subset(pca_ds, idxs)
    dl = torch.utils.data.DataLoader(subset_ds, args.infer_batch_size)

    pca_features = np.empty([num_images, full_features_dim])
    with torch.no_grad():
        for i, sequences in enumerate(tqdm(dl, ncols=100, desc="Database sequence descriptors for PCA: ")):
            if len(sequences.shape) == 5:
                sequences = einops.rearrange(sequences, "b s c h w -> (b s) c h w")
            features = model(sequences).cpu().numpy()
            pca_features[i * args.infer_batch_size : (i * args.infer_batch_size ) + len(features)] = features
    pca = PCA(args.pca_outdim)
    logging.info(f'Fitting PCA from {full_features_dim} to {args.pca_outdim}...')
    pca.fit(pca_features)
    return pca


def evaluation():
    args = parse_arguments()
    start_time = datetime.now()
    args.output_folder = f"test/{args.exp_name}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    setup_logging(args.output_folder, console="info")
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.output_folder}")

    ### Definition of the model
    model = TVGNet(args)

    if args.resume:
        state_dict = torch.load(args.resume)["model_state_dict"]
        state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
        model.load_state_dict(state_dict)

    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    meta = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    img_shape = (args.img_shape[0], args.img_shape[1])
    transform = configure_transform(image_dim=img_shape, meta=meta)

    eval_ds = BaseDataset(dataset_folder=args.dataset_path, split='test',
                          base_transform=transform, seq_len=args.seq_length,
                          pos_thresh=args.val_posDistThr, reverse_frames=args.reverse)
    logging.info(f"Test set: {eval_ds}")

    if args.pca_outdim:
        full_features_dim = args.features_dim
        args.features_dim = args.pca_outdim
        pca = compute_pca(args, model, transform, full_features_dim)
        model.module.meta['outputdim'] = args.pca_outdim
    else:
        pca = None

    logging.info(f"Output dimension of the model is {model.module.meta['outputdim']}")

    recalls, recalls_str = test(args, eval_ds, model, pca=pca)
    logging.info(f"Recalls on test set: {recalls_str}")
    logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")


if __name__ == "__main__":
    evaluation()
