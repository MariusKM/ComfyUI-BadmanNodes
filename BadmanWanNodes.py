import comfy.utils
import comfy.model_management
import comfy.clip_vision
import comfy.context_windows
import torch
import nodes
from node_helpers import conditioning_set_values


# Monkey patch to fix context window bug for WAN models with concat_latent_image
_original_get_resized_cond = comfy.context_windows.IndexListContextHandler.get_resized_cond

def _fixed_get_resized_cond(self, cond_in, x_in, window, device=None):
 
    if cond_in is None:
        return None
    
    resized_cond = []
    for actual_cond in cond_in:
        resized_actual_cond = actual_cond.copy()
        for key in actual_cond:
            try:
                cond_item = actual_cond[key]
                if isinstance(cond_item, torch.Tensor):
                    if self.dim < cond_item.ndim and cond_item.size(self.dim) == x_in.size(self.dim):
                        actual_cond_item = window.get_tensor(cond_item)
                        resized_actual_cond[key] = actual_cond_item.to(device)
                    else:
                        resized_actual_cond[key] = cond_item.to(device)
                elif key == "control":
                    resized_actual_cond[key] = self.prepare_control_objects(cond_item, device)
                elif isinstance(cond_item, dict):
                    new_cond_item = cond_item.copy()
                    for cond_key, cond_value in new_cond_item.items():
                        if isinstance(cond_value, torch.Tensor):
                            # FIX: Changed from cond_value.ndim < self.dim to self.dim < cond_value.ndim
                            # and from size(0) to size(self.dim) to match top-level tensor logic
                            if self.dim < cond_value.ndim and cond_value.size(self.dim) == x_in.size(self.dim):
                                new_cond_item[cond_key] = window.get_tensor(cond_value, device)
                            else:
                                new_cond_item[cond_key] = cond_value.to(device) if device else cond_value
                        elif hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
                            if self.dim < cond_value.cond.ndim and cond_value.cond.size(self.dim) == x_in.size(self.dim):
                                new_cond_item[cond_key] = cond_value._copy_with(window.get_tensor(cond_value.cond, device))
                        elif cond_key == "num_video_frames":
                            new_cond_item[cond_key] = cond_value._copy_with(cond_value.cond)
                            new_cond_item[cond_key].cond = window.context_length
                    resized_actual_cond[key] = new_cond_item
                else:
                    resized_actual_cond[key] = cond_item
            finally:
                del cond_item
        resized_cond.append(resized_actual_cond)
    return resized_cond

# Apply the patch
comfy.context_windows.IndexListContextHandler.get_resized_cond = _fixed_get_resized_cond


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
            # Repeat the last frame of start_image to fill the blend width region
            actual_frames = min(start_image.shape[0], length)
            blend_region_end = min(frame_blend_width, length)
            
            # Copy the actual frames we have
            image[:actual_frames] = start_image[:actual_frames]
            
            # If we have fewer frames than blend width, repeat the last frame
            if actual_frames < blend_region_end:
                image[actual_frames:blend_region_end] = start_image[actual_frames-1:actual_frames].expand(blend_region_end - actual_frames, -1, -1, -1)
            
            # Unmask the entire blend region
            mask[:, :, :blend_region_end + 3] = 0.0
        
        # Middle frame
        if middle_image is not None:
            middle_start = max(0, middle_frame_idx - frame_blend_width // 2)
            middle_end = min(length, middle_frame_idx + frame_blend_width // 2)
            blend_region_len = middle_end - middle_start
            actual_frames = min(middle_image.shape[0], blend_region_len)
            
            # Copy the actual frames we have
            image[middle_start:middle_start + actual_frames] = middle_image[:actual_frames]
            
            # If we have fewer frames than blend width, repeat the last frame
            if actual_frames < blend_region_len:
                image[middle_start + actual_frames:middle_end] = middle_image[actual_frames-1:actual_frames].expand(blend_region_len - actual_frames, -1, -1, -1)
            
            # Unmask the entire blend region
            mask[:, :, middle_start:min(length, middle_end + 3)] = 0.0
        
        # End frame at end
        if end_image is not None:
            end_start = max(0, length - frame_blend_width)
            blend_region_len = length - end_start
            actual_frames = min(end_image.shape[0], blend_region_len)
            
            # For end frame, we want to use the LAST frame(s) of the provided image
            # If end_image has multiple frames, use the last ones
            if end_image.shape[0] >= actual_frames:
                image[end_start:end_start + actual_frames] = end_image[-actual_frames:]
            else:
                # If we have fewer frames than needed, place what we have and repeat the last one
                image[end_start:end_start + end_image.shape[0]] = end_image
                if end_image.shape[0] < blend_region_len:
                    image[end_start + end_image.shape[0]:length] = end_image[-1:].expand(blend_region_len - end_image.shape[0], -1, -1, -1)
            
            # Unmask the entire blend region
            mask[:, :, end_start:length] = 0.0
        
        # Encode image to latent space
        concat_latent_image = vae.encode(image[:, :, :, :3])
        
        # Reshape mask to match latent dimensions with proper 4D structure for temporal processing
        # mask goes from (1, 1, T*4, H, W) -> (1, T, 4, H, W) -> (1, 4, T, H, W)
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        
        # Apply to conditioning
        # Note: When using context windows, these will be automatically subset by the context handler
        positive = conditioning_set_values(
            positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )
        negative = conditioning_set_values(
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
            positive = conditioning_set_values(
                positive, {"clip_vision_output": clip_vision_output}
            )
            negative = conditioning_set_values(
                negative, {"clip_vision_output": clip_vision_output}
            )
        
        # Return latent
        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)