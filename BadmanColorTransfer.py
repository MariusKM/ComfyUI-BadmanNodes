import numpy as np
import cv2
from PIL import Image
import torch
from skimage import color
from skimage.exposure import match_histograms, equalize_adapthist

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
                    "min": 0
                }),
                "method": (["original", "level_shift", "histogram", "adaptive"],),
                "preserve_details": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "mask": ("MASK",),
                "mask_threshold": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_lab_color_transfer"
    CATEGORY = "Badman"

    def create_mask(self, image, threshold=0.05):
        """
        Creates a mask excluding the black background.
        Args:
            image: RGB image in range [0, 1]
            threshold: Brightness threshold to separate foreground from background
        Returns:
            Binary mask where True indicates foreground
        """
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Create mask using threshold
        _, mask = cv2.threshold(gray, threshold * 255, 255, cv2.THRESH_BINARY)
        return mask > 0

    def prepare_mask(self, mask_input, image_shape, mask_threshold=0.05):
        """
        Prepares the mask either from input mask or generates it using threshold
        """
        if mask_input is not None:
            # Convert tensor mask to numpy if needed
            if isinstance(mask_input, torch.Tensor):
                mask = mask_input.cpu().numpy()
                if len(mask.shape) == 3 and mask.shape[0] == 1:  # Remove batch dimension
                    mask = np.squeeze(mask, axis=0)
            else:
                mask = mask_input
            
            # Ensure boolean type
            mask = mask > 0.5
        else:
            # Generate mask using threshold
            mask = self.create_mask(image_shape, mask_threshold)
        
        return mask

    def hex_to_rgb(self, hex_value):
        """
        Converts a hex integer (0xRRGGBB) to an (R, G, B) tuple.
        """
        r = (hex_value >> 16) & 0xFF
        g = (hex_value >> 8) & 0xFF
        b = hex_value & 0xFF
        return (r, g, b)

    def level_shift_lab(self, img_lab, target_lab, mask, preserve_details):
        """
        Adjusts the lightness channel while preserving details using level shifting.
        """
        L, A, B = cv2.split(img_lab)
        target_L = target_lab[0][0][0]
        
        # Calculate mean L only for masked region
        current_L = np.mean(L[mask]) if mask is not None else np.mean(L)
        
        # Calculate shift while considering detail preservation
        shift = (target_L - current_L) * (1 - preserve_details)
        
        # Apply shift while preserving relative differences
        L_adjusted = np.clip(L + shift, 0, 100).astype(np.uint8)
        
        # Ensure all channels have the same type and shape
        A = A.astype(np.uint8)
        B = B.astype(np.uint8)
        
        # Print debug information
        print(f"L shape: {L_adjusted.shape}, dtype: {L_adjusted.dtype}")
        print(f"A shape: {A.shape}, dtype: {A.dtype}")
        print(f"B shape: {B.shape}, dtype: {B.dtype}")
        
        # Create merged image
        try:
            return cv2.merge([L_adjusted, A, B])
        except Exception as e:
            print(f"Error during merge: {e}")
            print(f"Unique values in mask: {np.unique(mask)}")
            raise

    def histogram_match_lab(self, img_lab, target_lab, mask, preserve_details):
        """
        Adjusts the lightness channel using histogram matching.
        """
        L, A, B = cv2.split(img_lab)
        target_L = np.full_like(L, target_lab[0][0][0])
        
        if mask is not None:
            # Apply histogram matching only to masked region
            L_masked = L.copy()
            L_masked[~mask] = 0  # Set background to black
            L_matched = match_histograms(L_masked, target_L)
            L_matched[~mask] = L[~mask]  # Restore background
        else:
            L_matched = match_histograms(L, target_L)
        
        # Blend between original and matched histogram based on preserve_details
        L_adjusted = (L * preserve_details + L_matched * (1 - preserve_details)).astype(np.uint8)
        
        # Ensure all channels have the same type
        A = A.astype(np.uint8)
        B = B.astype(np.uint8)
        
        try:
            return cv2.merge([L_adjusted, A, B])
        except Exception as e:
            print(f"Error during merge: {e}")
            print(f"L shape: {L_adjusted.shape}, dtype: {L_adjusted.dtype}")
            print(f"A shape: {A.shape}, dtype: {A.dtype}")
            print(f"B shape: {B.shape}, dtype: {B.dtype}")
            raise

    def adaptive_scale_lab(self, img_lab, target_lab, mask, preserve_details):
        """
        Adjusts the lightness channel using adaptive scaling (CLAHE).
        """
        L, A, B = cv2.split(img_lab)
        
        # Normalize L channel to 0-1 range for CLAHE
        L_norm = L / 100.0
        
        if mask is not None:
            # Apply CLAHE only to masked region
            L_masked = L_norm.copy()
            L_masked[~mask] = 0  # Set background to black
        else:
            L_masked = L_norm

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        L_adapted = clahe.apply((L_masked * 255).astype(np.uint8))
        L_adapted = (L_adapted / 255.0) * 100

        if mask is not None:
            L_adapted[~mask] = L[~mask]  # Restore background
        
        # Blend between original and adapted based on preserve_details
        L_adjusted = (L * preserve_details + L_adapted * (1 - preserve_details)).astype(np.uint8)
        
        # Ensure all channels have the same type
        A = A.astype(np.uint8)
        B = B.astype(np.uint8)
        
        try:
            return cv2.merge([L_adjusted, A, B])
        except Exception as e:
            print(f"Error during merge: {e}")
            print(f"L shape: {L_adjusted.shape}, dtype: {L_adjusted.dtype}")
            print(f"A shape: {A.shape}, dtype: {A.dtype}")
            print(f"B shape: {B.shape}, dtype: {B.dtype}")
            raise

    def apply_lab_color_transfer(self, input_image, hex_color, method="original", preserve_details=0.5, mask=None, mask_threshold=0.05):
        """
        Applies color transfer using the specified method.
        """
        # Convert input to numpy array
        if isinstance(input_image, torch.Tensor):
            input_image_np = input_image.cpu().numpy()
        else:
            raise TypeError("Input image must be a PyTorch tensor")

        # Remove batch dimension if present
        if input_image_np.shape[0] == 1:
            input_image_np = np.squeeze(input_image_np, axis=0)

        # Extract alpha channel if it exists (assuming RGBA format)
        has_alpha = input_image_np.shape[-1] == 4
        if has_alpha:
            rgb = input_image_np[..., :3]
            alpha = input_image_np[..., 3]
        else:
            rgb = input_image_np

        # Prepare mask (either from input or generate using threshold)
        final_mask = self.prepare_mask(mask, rgb, mask_threshold)

        # Convert to uint8 format
        rgb_uint8 = (rgb * 255).astype(np.uint8)

        # Convert target color
        target_color = self.hex_to_rgb(hex_color)
        target_color_lab = cv2.cvtColor(np.uint8([[list(target_color)]]), cv2.COLOR_RGB2LAB)

        # Convert input image to Lab
        img_lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)

        # Apply the selected method
        if method == "level_shift":
            processed_lab = self.level_shift_lab(img_lab, target_color_lab, final_mask, preserve_details)
        elif method == "histogram":
            processed_lab = self.histogram_match_lab(img_lab, target_color_lab, final_mask, preserve_details)
        elif method == "adaptive":
            processed_lab = self.adaptive_scale_lab(img_lab, target_color_lab, final_mask, preserve_details)
        else:  # original method
            processed_lab = img_lab

        # Apply color transfer (A and B channels)
        L, A, B = cv2.split(processed_lab)
        target_A = target_color_lab[0][0][1]
        target_B = target_color_lab[0][0][2]
        
        # Only apply color to masked regions
        if final_mask is not None:
            A[final_mask] = target_A
            B[final_mask] = target_B
        else:
            A[:] = target_A
            B[:] = target_B
        
        # Merge channels
        processed_lab = cv2.merge([L, A, B])

        # Convert back to RGB
        recolored_rgb = cv2.cvtColor(processed_lab, cv2.COLOR_LAB2RGB)

        # Convert to float and ensure range 0-1
        recolored_rgb = recolored_rgb.astype(np.float32) / 255.0

        # Reconstruct the final image with alpha if needed
        if has_alpha:
            final_image = np.dstack((recolored_rgb, alpha))
        else:
            final_image = recolored_rgb

        # Convert to PyTorch tensor and add batch dimension
        final_tensor = torch.from_numpy(final_image).float().unsqueeze(0)

        return (final_tensor,)