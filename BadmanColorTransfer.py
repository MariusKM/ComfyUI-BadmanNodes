import numpy as np
import cv2
from PIL import Image
import torch

# Define the Lab Color Transfer Node
class LabColorTransferNode:
    def __init__(self, device="cpu"):
        self.device = device
    


    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "hex_color": ("INT", {
                    "default": 0,
                    "min": 0}),  # Expecting a color input in hexadecimal integer format
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_lab_color_transfer"
    CATEGORY = "Badman"

    def hex_to_rgb(self, hex_value):
        """
        Converts a hex integer (0xRRGGBB) to an (R, G, B) tuple.
        
        Parameters:
        - hex_value: A hex integer representing the color.
        
        Returns:
        - A tuple (R, G, B) with values in the range 0-255.
        """
        # Extract R, G, B components from the hex integer
        r = (hex_value >> 16) & 0xFF
        g = (hex_value >> 8) & 0xFF
        b = hex_value & 0xFF
        print(r,g,b)
        return (r, g, b)

    def apply_lab_color_transfer(self, input_image, hex_color):
        """
        Transfers the input color (from hex) to the image using Lab color transfer,
        preserving luminance and applying color transformation to A and B channels.
        """
        # Convert hex color to RGB tuple
        target_color = self.hex_to_rgb(hex_color)

        # Ensure the input is a PyTorch tensor, and convert to NumPy
        if isinstance(input_image, torch.Tensor):
            input_image_np = input_image.cpu().numpy()
        else:
            raise TypeError("Input image must be a PyTorch tensor")

        # Remove batch dimension if present (input shape is likely [1, H, W, 3])
        if input_image_np.shape[0] == 1:
            input_image_np = np.squeeze(input_image_np, axis=0)  # Remove the batch dimension

        # Now input_image_np should be in the format (H, W, 3) for RGB images
        # Ensure the image is in uint8 format (0-255 range)
        input_image_np = (input_image_np * 255).astype(np.uint8)

        # Convert the NumPy array (input image) to Lab color space using OpenCV
        img_lab = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2LAB)

        # Split the image into L, A, and B channels
        L_channel, A_channel, B_channel = cv2.split(img_lab)

        # Convert the target RGB color to Lab color space
        target_color_lab = cv2.cvtColor(np.uint8([[list(target_color)]]), cv2.COLOR_RGB2LAB)[0][0]
        target_A = target_color_lab[1]  # A component of target color
        target_B = target_color_lab[2]  # B component of target color

        # Replace the A and B channels of the image with the target A and B values
        A_channel[:] = target_A
        B_channel[:] = target_B

        # Merge the original L channel with the new A and B channels
        recolored_lab = cv2.merge([L_channel, A_channel, B_channel])

        # Convert the recolored Lab image back to RGB
        recolored_rgb = cv2.cvtColor(recolored_lab, cv2.COLOR_LAB2RGB)

        # Convert the result back to a PyTorch tensor
        # Convert the result back to a PyTorch tensor with the correct shape [batch, height, width, channels]
        recolored_rgb_tensor = torch.from_numpy(recolored_rgb / 255.0).float().unsqueeze(0)  # Add back batch dimension
        print(recolored_rgb_tensor.shape)
        # Return only a single image (combined RGB image)
        return (recolored_rgb_tensor,)