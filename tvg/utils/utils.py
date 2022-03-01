import shutil
from torchvision import transforms
import torch
import logging


def save_checkpoint(args, state, is_best, filename):
    model_path = f"{args.output_folder}/{filename}"
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, f"{args.output_folder}/best_model.pth")


def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    start_epoch_num = checkpoint["epoch_num"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_r5 = checkpoint["best_r5"]
    not_improved_num = checkpoint["not_improved_num"]
    logging.debug(f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, " \
                  f"current_best_R@5 = {best_r5:.1f}")
    if args.resume.endswith("last_model.pth"):  # Copy best model to current output_folder
        shutil.copy(args.resume.replace("last_model.pth", "best_model.pth"), args.output_folder)
    return model, optimizer, best_r5, start_epoch_num, not_improved_num


def load_pretrained_backbone(args, model):
    """Load a pretrained backbone"""
    logging.debug(f"Loading checkpoint: {args.pretrain_model}")
    checkpoint = torch.load(args.pretrain_model)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    return model


def configure_transform(image_dim, meta):
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    transform = transforms.Compose([
        transforms.Resize(image_dim),
        transforms.ToTensor(),
        normalize,
    ])

    return transform
