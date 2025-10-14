import comfy.utils
import comfy.model_management
import comfy.clip_vision
import torch
import nodes
from comfy.nodes import node_helpers


class WanThreeFrameToVideo:
    """
    Custom node that takes 3 keyframes (start, middle, end) and generates a video
    that transitions through all three frames with context window support.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4, "tooltip": "Total video length in frames"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "middle_frame_position": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Position of middle frame as fraction of total length (0.0-1.0)"}),
                "frame_blend_width": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1, "tooltip": "Number of frames to blend around each keyframe for smooth transitions"}),
            },
            "optional": {
                "start_image": ("IMAGE", ),
                "middle_image": ("IMAGE", ),
                "end_image": ("IMAGE", ),
                "clip_vision_start_image": ("CLIP_VISION_OUTPUT", ),
                "clip_vision_middle_image": ("CLIP_VISION_OUTPUT", ),
                "clip_vision_end_image": ("CLIP_VISION_OUTPUT", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    def execute(self, positive, negative, vae, width, height, length, batch_size, 
                middle_frame_position, frame_blend_width,
                start_image=None, middle_image=None, end_image=None, 
                clip_vision_start_image=None, clip_vision_middle_image=None, 
                clip_vision_end_image=None):
        
        spacial_scale = vae.spacial_compression_encode()
        latent = torch.zeros(
            [batch_size, vae.latent_channels, ((length - 1) // 4) + 1, 
             height // spacial_scale, width // spacial_scale], 
            device=comfy.model_management.intermediate_device()
        )
        
        # Initialize image with neutral gray and full mask (model will inpaint)
        image = torch.ones((length, height, width, 3)) * 0.5
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))
        
        # Upscale images to target resolution
        if start_image is not None:
            start_image = comfy.utils.common_upscale(
                start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        
        if middle_image is not None:
            middle_image = comfy.utils.common_upscale(
                middle_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        
        if end_image is not None:
            end_image = comfy.utils.common_upscale(
                end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        
        # Calculate keyframe positions
        middle_frame_idx = int(length * middle_frame_position)
        
        # Place keyframes with blend regions
        # Start frame at beginning
        if start_image is not None:
            start_end = min(start_image.shape[0], frame_blend_width)
            image[:start_end] = start_image[:start_end]
            mask[:, :, :start_end + 3] = 0.0
        
        # Middle frame
        if middle_image is not None:
            middle_start = max(0, middle_frame_idx - frame_blend_width // 2)
            middle_end = min(length, middle_frame_idx + frame_blend_width // 2)
            middle_img_len = min(middle_image.shape[0], middle_end - middle_start)
            image[middle_start:middle_start + middle_img_len] = middle_image[:middle_img_len]
            
            # Soft masking around middle frame (allow blending)
            blend_start = max(0, middle_frame_idx - frame_blend_width)
            blend_end = min(length, middle_frame_idx + frame_blend_width)
            mask[:, :, blend_start + 1:blend_end] = 0.0
        
        # End frame at end
        if end_image is not None:
            end_start = max(0, length - frame_blend_width)
            end_img_start = max(0, end_image.shape[0] - (length - end_start))
            image[end_start:] = end_image[end_img_start:]
            mask[:, :, -end_image.shape[0]:] = 0.0
        
        # Encode image to latent space
        concat_latent_image = vae.encode(image[:, :, :, :3])
        
        # Reshape mask to match latent dimensions
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        
        # Apply to conditioning
        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )
        
        # Handle clip vision outputs (concatenate all three if present)
        clip_vision_output = None
        
        if clip_vision_start_image is not None:
            clip_vision_output = clip_vision_start_image
        
        if clip_vision_middle_image is not None:
            if clip_vision_output is not None:
                states = torch.cat(
                    [clip_vision_output.penultimate_hidden_states, 
                     clip_vision_middle_image.penultimate_hidden_states], 
                    dim=-2
                )
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states
            else:
                clip_vision_output = clip_vision_middle_image
        
        if clip_vision_end_image is not None:
            if clip_vision_output is not None:
                states = torch.cat(
                    [clip_vision_output.penultimate_hidden_states, 
                     clip_vision_end_image.penultimate_hidden_states], 
                    dim=-2
                )
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states
            else:
                clip_vision_output = clip_vision_end_image
        
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(
                positive, {"clip_vision_output": clip_vision_output}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"clip_vision_output": clip_vision_output}
            )
        
        # Return latent
        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)