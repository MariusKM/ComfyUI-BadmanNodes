"""
BadmanWanOutpaintNodes.py
Custom nodes for calculating frame parameters for Wan 2.2 outpainting workflows.
"""


def is_valid_wan_frame_count(n):
    """
    Check if a frame count satisfies the WAN VAE encoding/decoding constraint.
    Valid frame counts follow the pattern: 1 + 4*k where k >= 0
    Examples: 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, ..., 81, 85, ...
    
    Args:
        n (int): Frame count to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    if n < 1:
        return False
    return (n - 1) % 4 == 0


def next_valid_wan_frame_count(n):
    """
    Round up to the next valid WAN frame count if the input is invalid.
    
    Args:
        n (int): Frame count to adjust
    
    Returns:
        int: Next valid frame count (>= n)
    """
    if n < 1:
        return 1
    if is_valid_wan_frame_count(n):
        return n
    # Round up to next valid: 1 + 4*ceil((n-1)/4)
    return 1 + 4 * ((n - 1 + 3) // 4)


def find_min_valid_context(min_context=9):
    """
    Find the minimum valid context frame count that is >= min_context
    and satisfies the VAE constraint.
    
    Args:
        min_context (int): Minimum desired context frames (default: 9)
    
    Returns:
        int: Minimum valid context frame count
    """
    return next_valid_wan_frame_count(min_context)


class WanOutpaintFrameCalculator:
    """
    Calculates frame parameters for multi-stage Wan 2.2 outpainting.
    
    Given a total video frame count, this node computes:
    - Number of sampler stages needed
    - Generation length for each sampler
    - Context frames for each sampler
    - Start position in source video for each sampler
    
    The calculations respect the WAN VAE encoding/decoding constraints where
    frame counts must follow the pattern: 1 + 4*k (k >= 0)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_frames": ("INT", {
                    "default": 192,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Total frame count of the input video"
                }),
            },
            "optional": {
                "additional_context_per_sampler": ("STRING", {
                    "default": "0,0,0",
                    "multiline": False,
                    "tooltip": "Comma-separated list of additional context frames per sampler (e.g., '0,0,4' adds 4 extra context frames to the 3rd sampler). Values shorter than samplers count will be padded with 0."
                }),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("num_samplers", "generation_lengths", "context_frames", "start_positions")
    FUNCTION = "calculate"
    CATEGORY = "video/wan"
    
    def calculate(self, total_frames, additional_context_per_sampler="0,0,0"):
        """
        Calculate frame parameters for Wan outpainting.
        
        Args:
            total_frames (int): Total frame count of input video
            additional_context_per_sampler (str): Comma-separated additional context values
        
        Returns:
            tuple: (num_samplers, generation_lengths, context_frames, start_positions)
        """
        # Parse additional context input
        additional_context = []
        if additional_context_per_sampler and additional_context_per_sampler.strip():
            try:
                additional_context = [int(x.strip()) for x in additional_context_per_sampler.split(',') if x.strip()]
            except ValueError:
                # If parsing fails, default to no additional context
                additional_context = []
        
        # The input total_frames can be any value - we'll break it down into valid chunks
        # Only adjust if it would cause frame loss in a single-sampler scenario
        
        # If video is 81 frames or less, only need 1 sampler
        # In this case, adjust to valid frame count to avoid loss
        if total_frames <= 81:
            adjusted_frames = next_valid_wan_frame_count(total_frames)
            return (1, [adjusted_frames], [0], [0])
        
        # Initialize lists for multi-sampler case
        generation_lengths = []
        context_frames = []
        start_positions = []
        
        # First sampler always generates 81 frames with no context
        generation_lengths.append(81)
        context_frames.append(0)
        start_positions.append(0)
        
        # Track how many frames we've covered (after first generation)
        frames_covered = 81
        
        # Calculate remaining samplers
        sampler_index = 1  # Start at 1 since sampler 0 is already done
        while frames_covered < total_frames:
            remaining_frames = total_frames - frames_covered
            
            # Get additional context for this sampler (0 if not specified)
            extra_context = additional_context[sampler_index] if sampler_index < len(additional_context) else 0
            extra_context = max(0, extra_context)  # Ensure non-negative
            
            # Determine context frames (minimum valid >= 9) + additional context
            base_min_context = find_min_valid_context(9)
            desired_context = base_min_context + extra_context
            # Ensure the desired context is also a valid frame count
            min_context = next_valid_wan_frame_count(desired_context)
            
            # Determine generation length for this sampler
            if remaining_frames <= 81 - min_context:
                # This is the last sampler
                # We need to generate: remaining_frames + context_frames
                # And the result must be a valid WAN frame count
                gen_length = next_valid_wan_frame_count(remaining_frames + min_context)
                # Context is whatever makes the math work out
                ctx_frames = gen_length - remaining_frames
            else:
                # Not the last sampler - generate 81 frames with minimum context
                gen_length = 81
                ctx_frames = min_context
            
            generation_lengths.append(gen_length)
            context_frames.append(ctx_frames)
            start_positions.append(frames_covered)
            
            # Update frames covered (subtract context since those frames overlap)
            frames_covered += (gen_length - ctx_frames)
            sampler_index += 1
        
        num_samplers = len(generation_lengths)
        
        return (num_samplers, generation_lengths, context_frames, start_positions)


# Node exports
NODE_CLASS_MAPPINGS = {
    "BadmanWanOutpaintFrameCalculator": WanOutpaintFrameCalculator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BadmanWanOutpaintFrameCalculator": "WAN Outpaint Frame Calculator (Badman)",
}

