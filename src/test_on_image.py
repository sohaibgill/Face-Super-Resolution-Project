import torch
import argparse
import os
from torchvision import transforms
from PIL import Image
from collections import OrderedDict
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.nn as nn
from model import RRDBNet_arch as RRDBNet_arch
import numpy as np


def load_network(load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)


_transform = transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                      std=[0.5, 0.5, 0.5])])


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, default="../test_images/02220.png", help="Path to image")
parser.add_argument("--checkpoint_model", type=str, default="../Checkpoints/generator_5.pth", help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")

opt = parser.parse_args()

os.makedirs("../Results/testing", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = RRDBNet_arch.RRDBNet(in_nc=opt.channels, out_nc=3, nf=64, nb=16).to(device)
generator = DataParallel(generator)
load_network(opt.checkpoint_model, generator, True)

generator.eval()

# Prepare input
image_tensor = torch.unsqueeze(_transform(Image.open(opt.image_path).convert('RGB')), 0).to(device)
# Upsample image
with torch.no_grad():
    sr_image = generator(image_tensor).squeeze(0).cpu().numpy()
    print(sr_image.shape)
    sr_image = np.clip((np.transpose(sr_image, (1, 2, 0)) / 2.0 + 0.5) * 255.0, 0, 255).astype(np.uint8)

# Save image
fn = opt.image_path.split("/")[-1]
save_image(sr_image, f"../Results/testing/sr-{fn}")
