import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image
import math
import torchvision.transforms.functional as Ft
import comfy.utils
import comfy.model_management
import copy
import cv2


#Code for the following two nodes taken from https://github.com/cubiq/ComfyUI_essentials.git
class ImageDesaturate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "factor": ("FLOAT", { "default": 1.00, "min": 0.00, "max": 1.00, "step": 0.05, }),
                "method": (["luminance (Rec.709)", "luminance (Rec.601)", "average", "lightness"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Badman"

    def execute(self, image, factor, method):
        if method == "luminance (Rec.709)":
            grayscale = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
        elif method == "luminance (Rec.601)":
            grayscale = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        elif method == "average":
            grayscale = image.mean(dim=3)
        elif method == "lightness":
            grayscale = (torch.max(image, dim=3)[0] + torch.min(image, dim=3)[0]) / 2

        grayscale = (1.0 - factor) * image + factor * grayscale.unsqueeze(-1).repeat(1, 1, 1, 3)
        grayscale = torch.clamp(grayscale, 0, 1)

        return(grayscale,)

class MaskBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "amount": ("INT", { "default": 6, "min": 0, "max": 256, "step": 1, }),
                "device": (["auto", "cpu", "gpu"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "Badman"

    def execute(self, mask, amount, device):
        if amount == 0:
            return (mask,)

        if "gpu" == device:
            mask = mask.to(comfy.model_management.get_torch_device())
        elif "cpu" == device:
            mask = mask.to('cpu')

        if amount % 2 == 0:
            amount+= 1

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        mask = T.functional.gaussian_blur(mask.unsqueeze(1), amount).squeeze(1)

        if "gpu" == device or "cpu" == device:
            mask = mask.to(comfy.model_management.intermediate_device())

        return(mask,)


class DilateErodeMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks": ("MASK",),
                "radius": ("INT", {
                    "default": 0,
                    "min": -1023,
                    "max": 1023,
                    "step": 1
                }),
                "shape": (["box", "circle"],),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "dilate_mask"

    CATEGORY = "Badman"

    def dilate_mask(self, masks, radius, shape):
        
        if radius == 0:
            return (masks,)
        
        s = abs(radius)
        d = s * 2 + 1
        k = np.zeros((d, d), np.uint8)
        if shape == "circle":
            k = cv2.circle(k, (s,s), s, 1, -1)
        else:
            k += 1
        
        dup = copy.deepcopy(masks.cpu().numpy())
        
        for index, mask in enumerate(dup):
            if radius > 0:
                dup[index] = cv2.dilate(mask, k, iterations=1)
            else:
                dup[index] = cv2.erode(mask, k, iterations=1)
        
        return (torch.from_numpy(dup),)

class Blend:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "blend_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light", "difference", "add"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_images"

    CATEGORY = "Badman"

    def blend_images(self, image1: torch.Tensor, image2: torch.Tensor, blend_factor: float, blend_mode: str):
        image2 = image2.to(image1.device)
        if image1.shape != image2.shape:
            image2 = image2.permute(0, 3, 1, 2)
            image2 = comfy.utils.common_upscale(image2, image1.shape[2], image1.shape[1], upscale_method='bicubic', crop='center')
            image2 = image2.permute(0, 2, 3, 1)

        blended_image = self.blend_mode(image1, image2, blend_mode)
        blended_image = image1 * (1 - blend_factor) + blended_image * blend_factor
        blended_image = torch.clamp(blended_image, 0, 1)
        return (blended_image,)

    def blend_mode(self, img1, img2, mode):
        if mode == "normal":
            return img2
        elif mode == "multiply":
            return img1 * img2
        elif mode == "add":
            return img1 + img2
        elif mode == "screen":
            return 1 - (1 - img1) * (1 - img2)
        elif mode == "overlay":
            return torch.where(img1 <= 0.5, 2 * img1 * img2, 1 - 2 * (1 - img1) * (1 - img2))
        elif mode == "soft_light":
            return torch.where(img2 <= 0.5, img1 - (1 - 2 * img2) * img1 * (1 - img1), img1 + (2 * img2 - 1) * (self.g(img1) - img1))
        elif mode == "difference":
            return img1 - img2
        else:
            raise ValueError(f"Unsupported blend mode: {mode}")

    def g(self, x):
        return torch.where(x <= 0.25, ((16 * x - 12) * x + 4) * x, torch.sqrt(x))

class HexGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "r": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "g": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "b": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "grayscale": ("BOOLEAN",),
                
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "linear_rgb_to_int"

    CATEGORY = "Badman"

    def linear_rgb_to_int(self,r, g, b, grayscale=False):
        """
        Converts linear RGB values to an integer color code.

        Parameters:
        r (float): Red value (0.0-1.0)
        g (float): Green value (0.0-1.0)
        b (float): Blue value (0.0-1.0)
        grayscale (bool): If True, use the r value for all RGB components

        Returns:
        int: Integer color code in the format 0xRRGGBB
        """
    
        if grayscale:
            g = b = r
        
        # Convert float values to int (0-255)
        r_int = int(round(r * 255))
        g_int = int(round(g * 255))
        b_int = int(round(b * 255))
        
        # Combine into a single integer
        color_int = (r_int << 16) + (g_int << 8) + b_int
        return (color_int,)
    
# Taken from Yancs Node pack https://github.com/ALatentPlace/ComfyUI_yanc
def permute_tt(image):
    return image.permute(0, 3, 1, 2)

def permute_ft(image):
    return image.permute(0, 2, 3, 1)

class Brightness:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {
                    "image": ("IMAGE",),
                    "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                },
                "optional":
                {
                    "mask_opt": ("MASK",),
                }
                }

    CATEGORY = "Badman"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "do_it"

    def do_it(self, image, brightness, mask_opt=None):

        if mask_opt is not None:
            mask = mask_opt.clone()
            mask = permute_tt(mask.unsqueeze(-1))
        else:
            mask = torch.ones_like(image)
            mask = permute_tt(mask)

        img = image.clone()
        img = permute_tt(img)
        img = Ft.adjust_brightness(img * mask, brightness)
        img = img + permute_tt(image) * Ft.invert(mask)
        img = permute_ft(img)

        return (img,)


# ------------------------------------------------------------------------------------------------------------------ #

import torch
import math
import random
import time

class RandomColorImageGrid:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 1}),
                "height": ("INT", {"default": 1024, "min": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "num_colors": ("INT", {"default": 4, "min": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"

    CATEGORY = "image"

    def generate(self, width, height, batch_size=1, num_colors=4):
        # Seed the random number generator uniquely for each call
        random.seed(time.time() + random.randint(0, 10000))

        # Calculate rows and columns based on number of colors
        rows = math.ceil(math.sqrt(num_colors))
        cols = math.ceil(num_colors / rows)

        tile_width = width // cols
        tile_height = height // rows

        # Create tensors for the R, G, B channels
        images = []
        for _ in range(batch_size):
            r = torch.zeros([height, width], dtype=torch.float32, device=self.device)
            g = torch.zeros([height, width], dtype=torch.float32, device=self.device)
            b = torch.zeros([height, width], dtype=torch.float32, device=self.device)

            # Generate random colors and fill the tiles
            color_idx = 0
            for i in range(rows):
                for j in range(cols):
                    if color_idx >= num_colors:
                        break
                    color_r = random.randint(0, 255) / 255.0
                    color_g = random.randint(0, 255) / 255.0
                    color_b = random.randint(0, 255) / 255.0

                    x_start, x_end = j * tile_width, (j + 1) * tile_width
                    y_start, y_end = i * tile_height, (i + 1) * tile_height

                    r[y_start:y_end, x_start:x_end] = color_r
                    g[y_start:y_end, x_start:x_end] = color_g
                    b[y_start:y_end, x_start:x_end] = color_b

                    color_idx += 1

            # Concatenate the R, G, B channels along the last dimension
            image = torch.stack([r, g, b], dim=-1)
            images.append(image)

        # Return the batch of images
        return (torch.stack(images),)
