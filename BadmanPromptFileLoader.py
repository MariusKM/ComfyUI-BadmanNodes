import os
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence


def parse_prompt_file(text, delimiter="---"):
    """Parse a prompt file with the structure:
        ---
        filename.png
        ---
        prompt text (may span multiple lines)

        ---
        filename2.png
        ---
        prompt text2
    """
    lines = text.splitlines()
    entries = []
    i = 0
    n = len(lines)
    while i < n:
        while i < n and lines[i].strip() != delimiter:
            i += 1
        if i >= n:
            break
        i += 1
        while i < n and lines[i].strip() == "":
            i += 1
        if i >= n:
            break
        filename = lines[i].strip()
        i += 1
        while i < n and lines[i].strip() != delimiter:
            i += 1
        if i >= n:
            break
        i += 1
        prompt_lines = []
        while i < n and lines[i].strip() != delimiter:
            prompt_lines.append(lines[i])
            i += 1
        prompt = "\n".join(prompt_lines).strip()
        if filename:
            entries.append((filename, prompt))
    return entries


def load_image_as_tensor(path):
    img = Image.open(path)
    output_images = []
    output_masks = []
    for frame in ImageSequence.Iterator(img):
        frame = ImageOps.exif_transpose(frame)
        if frame.mode == "I":
            frame = frame.point(lambda i: i * (1 / 255))
        rgb = frame.convert("RGB")
        arr = np.array(rgb).astype(np.float32) / 255.0
        output_images.append(torch.from_numpy(arr)[None,])
        if "A" in frame.getbands():
            mask = np.array(frame.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((arr.shape[0], arr.shape[1]), dtype=torch.float32)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1:
        image = torch.cat(output_images, dim=0)
        mask = torch.cat(output_masks, dim=0)
    else:
        image = output_images[0]
        mask = output_masks[0]
    return image, mask


class BadmanPromptFileImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_directory": ("STRING", {"default": "", "multiline": False}),
                "prompt_file": ("STRING", {"default": "", "multiline": False}),
                "index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "delimiter": ("STRING", {"default": "---", "multiline": False}),
                "wrap_index": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "INT")
    RETURN_NAMES = ("image", "mask", "prompt", "filename", "total")
    FUNCTION = "load"
    CATEGORY = "Badman"

    @classmethod
    def IS_CHANGED(cls, image_directory, prompt_file, index, delimiter, wrap_index):
        try:
            mtime = os.path.getmtime(prompt_file)
        except OSError:
            mtime = 0
        return f"{prompt_file}|{mtime}|{index}|{delimiter}|{wrap_index}|{image_directory}"

    def load(self, image_directory, prompt_file, index, delimiter, wrap_index):
        if not prompt_file or not os.path.isfile(prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        if not image_directory or not os.path.isdir(image_directory):
            raise NotADirectoryError(f"Image directory not found: {image_directory}")

        with open(prompt_file, "r", encoding="utf-8") as f:
            text = f.read()

        entries = parse_prompt_file(text, delimiter=delimiter)
        total = len(entries)
        if total == 0:
            raise ValueError(f"No entries parsed from prompt file: {prompt_file}")

        if wrap_index:
            idx = index % total
        else:
            if index < 0 or index >= total:
                raise IndexError(f"Index {index} out of range (0..{total - 1})")
            idx = index

        filename, prompt = entries[idx]
        image_path = os.path.join(image_directory, filename)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found for entry {idx}: {image_path}")

        image, mask = load_image_as_tensor(image_path)
        return (image, mask, prompt, filename, total)


NODE_CLASS_MAPPINGS = {
    "BadmanPromptFileImageLoader": BadmanPromptFileImageLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BadmanPromptFileImageLoader": "Prompt File Image Loader (Badman)",
}
