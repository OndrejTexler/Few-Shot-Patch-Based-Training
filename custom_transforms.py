import numpy as np
from torchvision import transforms
from scipy import ndimage
import torch


def to_image_space(x):
    return ((np.clip(x, -1, 1) + 1) / 2 * 255).astype(np.uint8)


def to_rgb(x):
    return x if x.mode == 'RGB' else x.convert('RGB')


def to_l(x):
    return x if x.mode == 'L' else x.convert('L')


def blur_mask(tensor):
    np_tensor = tensor.numpy()
    smoothed = ndimage.gaussian_filter(np_tensor, sigma=20)
    return torch.FloatTensor(smoothed)


def build_transform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), mask=False):
    #if type(image_size) != tuple:
        #image_size = (image_size, image_size)
    t = [#transforms.Resize((image_size[0], image_size[1])),
         to_rgb,
         transforms.ToTensor(),
         transforms.Normalize(mean, std)]
    if mask:
        t.append(blur_mask)
    return transforms.Compose(t)


def build_mask_transform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    t = [#transforms.Resize((image_size, image_size)),
         to_l,
         transforms.ToTensor()]
    return transforms.Compose(t)


def to_pil(tensor):
    t = transforms.ToPILImage()
    return t(tensor)


def tensor_mb(tensor):
    return (tensor.element_size() * tensor.nelement()) / 1024 / 1024






