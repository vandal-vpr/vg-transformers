import torch
import logging
import torchvision
from torch import nn
import einops
import re
# Transformers libraries
from timesformer.models.vit import TimeSformer as OfficialTimeSformer
from transformers import ViTModel
# our imports
from tvg.models import pooling
from tvg.models.normalize import L2Norm
from tvg.models.cct import cct_14_7x2_224, cct_14_7x2_384


class TVGNet(nn.Module):
    """Network for VG applied to sequences.The used networks are composed of 
    an encoder, a pooling layer and an aggregator."""
    def __init__(self, args):
        super().__init__()
        self.fusion = ''
        if args.arch in ['timesformer']:
            self.fusion = 'early'
        elif args.aggregation in ['seqvlad', 'fc']:
            self.fusion = 'intermediate'
        else:
            # it's cat
            self.fusion = 'late'
        self.arch = args.arch
        self.aggregator_name = args.aggregation
        # build model by coupling encoding - pooling - aggregation
        self.encoder = get_encoder(args)
        self.pooling = get_pooling(args)
        self.aggregator = get_aggregator(args)

        self.train_batch_size = args.train_batch_size
        self.seq_length = args.seq_length
        self.img_shape = args.img_shape
        self.stack_frames_features = lambda x: einops.rearrange(x, '(b s) d -> b (s d)', s=self.seq_length)
        if self.aggregator_name == 'fc':
            self.meta = {'outputdim': args.fc_out}
        else:
            self.meta = {'outputdim': args.features_dim}

    def forward(self, x):
        if self.fusion == 'early':
            x = einops.rearrange(x, '(b s) c h w -> b c s h w', s=self.seq_length)
            x = self.encoder(x)
            return x

        x = self.encoder(x)
        if self.arch == 'vit':
            if self.aggregator_name != 'seqvlad':
                # take only CLS token
                x = x.last_hidden_state[:, 0, :]
            else:
                # take all tokens but the CLS
                x = x.last_hidden_state[:, 1:, :]

        x = self.pooling(x)
        if self.fusion == 'intermediate':
            if self.aggregator_name == 'fc':
                x = self.stack_frames_features(x)

            x = self.aggregator(x)

        elif self.fusion == 'late':
            x = self.stack_frames_features(x)
        return x


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1

        return x[:, :, 0, 0]


def get_aggregator(args):
    if args.aggregation is None:
        raise ValueError("you should specify an aggregation method in args")

    if args.pooling == "netvlad":
        args.features_dim *= 64

    if args.aggregation == "seqvlad":
        is_transf_backbone = args.arch.startswith('cct') or args.arch.startswith('vit')
        aggregator = pooling.SeqVLAD(seq_length=args.seq_length, dim=args.features_dim,
                                     transf_backbone=is_transf_backbone)
        args.features_dim *= 64
        return aggregator

    elif args.aggregation == 'cat':
        args.features_dim *= args.seq_length
        return nn.Identity()

    elif args.aggregation == 'fc':
        fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(args.features_dim*args.seq_length, args.fc_out)
        )
        args.features_dim = args.fc_out
        return fc


def get_pooling(args):
    if args.pooling == "gem":
        return nn.Sequential(L2Norm(), pooling.GeM(), Flatten())
    elif args.pooling == "spoc":
        return nn.Sequential(L2Norm(), pooling.SPoC(), Flatten())
    elif args.pooling == "mac":
        return nn.Sequential(L2Norm(), pooling.MAC(), Flatten())
    elif args.pooling == "rmac":
        return nn.Sequential(L2Norm(), pooling.RMAC(), Flatten())
    elif args.pooling == "netvlad":
        return pooling.NetVLAD(dim=args.features_dim)
    elif args.pooling == "none" or args.pooling == '_':
        return nn.Identity()


def get_encoder(args):
    if args.arch.startswith("vit"):
        encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        if args.trunc_te:
            logging.debug(f"Truncate ViT at transformers encoder {args.trunc_te}")
            encoder.encoder.layer = encoder.encoder.layer[:args.trunc_te]
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te}")
            for p in encoder.parameters():
                p.requires_grad = False
            for name, child in encoder.encoder.layer.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 768
        return encoder
    elif args.arch.startswith("cct"):
        use_all_tokens = args.aggregation in ['cat', 'fc']
        if args.arch.startswith("cct224"):
            encoder = cct_14_7x2_224(pretrained=True, progress=True, use_all_tokens=use_all_tokens)
        elif args.arch.startswith("cct384"):
            encoder = cct_14_7x2_384(pretrained=True, progress=True, use_all_tokens=use_all_tokens)
        if args.trunc_te:
            logging.debug(f"Truncate CCT at transformers encoder {args.trunc_te}")
            encoder.classifier.blocks = torch.nn.ModuleList(encoder.classifier.blocks[:args.trunc_te].children())
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te}")
            for p in encoder.parameters():
                p.requires_grad = False
            for name, child in encoder.classifier.blocks.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 384
        return encoder
    elif args.arch == 'timesformer':
        assert args.img_shape[0] == args.img_shape[1]
        pre_trained = '' if args.pretrain_model is None else args.pretrain_model
        encoder = OfficialTimeSformer(img_size=args.img_shape[0],
                                      num_classes=0,
                                      num_frames=args.seq_length,
                                      attention_type='divided_space_time',
                                      pretrained_model=pre_trained)
        if args.trunc_te:
            logging.debug(f"Truncate Timesformer at encoder block {args.trunc_te}")
            encoder.model.blocks = torch.nn.ModuleList(encoder.model.blocks[:args.trunc_te].children())
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to Timesformer encoder block {args.freeze_te}")
            for p in encoder.parameters():
                p.requires_grad = False
            for name, child in encoder.model.blocks.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 768
        return encoder
    elif args.arch == 'vgg16':
        encoder = torchvision.models.vgg16(pretrained=True)
        layers = list(encoder.features.children())[:-2]
        for l in layers[:-5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the vgg16, freeze the previous ones")
        backbone = torch.nn.Sequential(*layers)
        args.features_dim = get_output_channels_dim(backbone) # Dinamically obtain number of channels in output
        return backbone
    elif args.arch.startswith("r"):  # It's a ResNet
        if args.arch.startswith("r18"):
            encoder = torchvision.models.resnet18(pretrained=True)
        elif args.arch.startswith("r50"):
            encoder = torchvision.models.resnet50(pretrained=True)
        elif args.arch.startswith("r101"):
            encoder = torchvision.models.resnet101(pretrained=True)
        for name, child in encoder.named_children():
            if name == args.freeze_layer:  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        if args.arch.endswith("l3"):
            logging.debug(f"Train only layer3 of the resnet{args.arch[1:3]} (remove layer4), freeze the previous ones")
            layers = list(encoder.children())[:-3]
        elif args.arch.endswith("l4"):
            logging.debug(f"Train only layer3 and layer4 of the resnet{args.arch[1:3]}, freeze the previous ones")
            layers = list(encoder.children())[:-2]
    
        encoder = torch.nn.Sequential(*layers)
        args.features_dim = get_output_channels_dim(encoder)  # Dinamically obtain number of channels in output
        return encoder


def get_output_channels_dim(model):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 3, 224, 224])).shape[1]
