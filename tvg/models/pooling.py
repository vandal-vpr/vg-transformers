import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import einops

from ..models import functionals as LF

import math
import logging
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import faiss
from transformers import ViTModel
from tqdm import tqdm


class MAC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return LF.mac(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SPoC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return LF.spoc(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return LF.gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class RMAC(nn.Module):
    def __init__(self, L=3, eps=1e-6):
        super().__init__()
        self.L = L
        self.eps = eps
    
    def forward(self, x):
        return LF.rmac(x, L=self.L, eps=self.eps)
    
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'L=' + '{}'.format(self.L) + ')'


# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, clusters_num=64, dim=128, normalize_input=True):
        """
        Args:
            clusters_num : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.clusters_num = clusters_num
        self.dim = dim
        self.alpha = 0
        self.normalize_input = normalize_input
        self.features_dim = dim * clusters_num
        self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(clusters_num, dim))

    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
        self.centroids = nn.Parameter(torch.from_numpy(centroids))
        self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x):
        N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = torch.zeros([N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device)
        for D in range(self.clusters_num):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                       self.centroids[D:D + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:, D:D + 1, :].unsqueeze(2)
            vlad[:, D:D + 1, :] = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

    def initialize_netvlad_layer(self, args, cluster_ds, encoder, save_to_file=False):
        descriptors_num = 50000
        descs_num_per_image = 100
        images_num = math.ceil(descriptors_num / descs_num_per_image)
        random_images = np.random.choice(cluster_ds.queries_num + cluster_ds.database_num, images_num, replace=False)
        subset_ds = Subset(cluster_ds, random_images)
        loader = DataLoader(dataset=subset_ds, num_workers=4,
                            batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))

        features_dim = self.features_dim // 64
        with torch.no_grad():
            encoder = encoder.eval()
            logging.debug("Extracting features to initialize NetVLAD layer")
            descriptors = np.zeros(shape=(descriptors_num, features_dim), dtype=np.float32)
            for iteration, (inputs, _, _) in enumerate(tqdm(loader, ncols=100)):
                inputs = inputs.to(args.device).view(-1, 3, args.img_shape[0], args.img_shape[1])
                inputs = inputs[::args.seq_length] # take only first frame of each sequence
                outputs = encoder(inputs)
                norm_outputs = F.normalize(outputs, p=2, dim=1)
                image_descriptors = norm_outputs.view(norm_outputs.shape[0], features_dim, -1).permute(0, 2, 1)
                image_descriptors = image_descriptors.cpu().numpy()
                batchix = iteration * args.infer_batch_size * descs_num_per_image
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(image_descriptors.shape[1], descs_num_per_image, replace=False)
                    startix = batchix + ix * descs_num_per_image
                    descriptors[startix:startix + descs_num_per_image, :] = image_descriptors[ix, sample, :]
        kmeans = faiss.Kmeans(features_dim, 64, niter=100, verbose=False)
        kmeans.train(descriptors)
        logging.debug(f"NetVLAD centroids shape: {kmeans.centroids.shape}")
        # if save_to_file: # or args.save_centroids:
        #     self.save_centroids(kmeans.centroids, descriptors)
        self.init_params(kmeans.centroids, descriptors)
        self = self.to(args.device)

    # @staticmethod
    # def save_centroids(centroids, descriptors, filename="test.hdf5"):
    #     print('====> Storing centroids', centroids.shape)
    #     with h5py.File(filename, mode='w') as h5:
    #         dbFeat = h5.create_dataset("descriptors", data=descriptors)
    #         h5.create_dataset('centroids', data=centroids)

# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class SeqVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, seq_length, clusters_num=64, dim=128, normalize_input=True, transf_backbone=False):
        """
        Args:
            clusters_num : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.clusters_num = clusters_num
        self.dim = dim
        self.alpha = 0
        self.normalize_input = normalize_input
        self.features_dim = dim * clusters_num
        self.transf_backbone = transf_backbone
        if transf_backbone:
            self.conv = nn.Conv1d(dim, clusters_num, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(clusters_num, dim))
        self.seq_length = seq_length

    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
        self.centroids = nn.Parameter(torch.from_numpy(centroids))
        if self.transf_backbone:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2))
        else:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x):
        if self.transf_backbone:
            x = einops.rearrange(x, '(b s) d c -> b c (s d)', s=self.seq_length)
            N, D, _ = x.shape[:]
        else:
            x = einops.rearrange(x, '(b s) c h w -> b c (s h) w', s=self.seq_length)
            N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = torch.zeros([N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device)
        for D in range(self.clusters_num):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                       self.centroids[D:D + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:, D:D + 1, :].unsqueeze(2)
            vlad[:, D:D + 1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

    def initialize_seqvlad_layer(self, args, cluster_ds, encoder, save_to_file=False):
        descriptors_num = 50000
        descs_num_per_image = 100
        images_num = math.ceil(descriptors_num / descs_num_per_image)
        random_images = np.random.choice(cluster_ds.queries_num + cluster_ds.database_num, images_num, replace=False)
        subset_ds = Subset(cluster_ds, random_images)
        loader = DataLoader(dataset=subset_ds, num_workers=4,
                            batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))

        features_dim = self.features_dim // 64
        with torch.no_grad():
            encoder = encoder.eval()
            logging.debug("Extracting features to initialize SeqVLAD layer")
            descriptors = np.zeros(shape=(descriptors_num, features_dim), dtype=np.float32)
            for iteration, (inputs, _, _) in enumerate(tqdm(loader, ncols=100)):
                inputs = inputs.to(args.device).view(-1, 3, args.img_shape[0], args.img_shape[1])
                inputs = inputs[::args.seq_length]  # take only first frame of each sequence
                outputs = encoder(inputs)
                if isinstance(encoder, ViTModel):
                    outputs = outputs.last_hidden_state[:, 1:, :]
                if self.transf_backbone:
                    # if using a transformer backbone, normalization is done token-wise
                    norm_outputs = F.normalize(outputs, p=2, dim=2)
                else:
                    norm_outputs = F.normalize(outputs, p=2, dim=1)

                image_descriptors = norm_outputs.view(norm_outputs.shape[0], features_dim, -1).permute(0, 2, 1)
                image_descriptors = image_descriptors.cpu().numpy()
                batchix = iteration * args.infer_batch_size * descs_num_per_image
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(image_descriptors.shape[1], descs_num_per_image, replace=False)
                    startix = batchix + ix * descs_num_per_image
                    descriptors[startix:startix + descs_num_per_image, :] = image_descriptors[ix, sample, :]
        kmeans = faiss.Kmeans(features_dim, 64, niter=100, verbose=False)
        kmeans.train(descriptors)
        logging.debug(f"SeqVLAD centroids shape: {kmeans.centroids.shape}")
        # if save_to_file: # or args.save_centroids:
        #     self.save_centroids(kmeans.centroids, descriptors)
        self.init_params(kmeans.centroids, descriptors)
        self = self.to(args.device)

    # @staticmethod
    # def save_centroids(centroids, descriptors, filename="test.hdf5"):
    #     print('====> Storing centroids', centroids.shape)
    #     with h5py.File(filename, mode='w') as h5:
    #         dbFeat = h5.create_dataset("descriptors", data=descriptors)
    #         h5.create_dataset('centroids', data=centroids)