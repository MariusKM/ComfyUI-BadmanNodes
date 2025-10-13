import torch
import torch.nn.functional as F


class InjectLatentNoiseMasked:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "latent": ("LATENT", ),
                    "mask": ("MASK", ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "noise_strength": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step":0.01, "round": 0.01}),
                    "normalize": (["false", "true"], {"default": "false"}),
                    "blend_mode": (["replace", "add", "multiply"], {"default": "replace"}),
                }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"
    CATEGORY = "Badman"

    def execute(self, latent, mask, noise_seed, noise_strength, normalize="false", blend_mode="replace"):
        torch.manual_seed(noise_seed)
        noise_latent = latent.copy()
        original_samples = noise_latent["samples"].clone()
        random_noise = torch.randn_like(original_samples)

        if normalize == "true":
            mean = original_samples.mean()
            std = original_samples.std()
            random_noise = random_noise * std + mean

        # Prepare mask to match latent dimensions
        mask = F.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), 
                            size=(original_samples.shape[2], original_samples.shape[3]), 
                            mode="bilinear")
        mask = mask.expand((-1, original_samples.shape[1], -1, -1)).clamp(0.0, 1.0)
        
        # Handle batch size mismatches
        if mask.shape[0] < original_samples.shape[0]:
            mask = mask.repeat((original_samples.shape[0] - 1) // mask.shape[0] + 1, 1, 1, 1)[:original_samples.shape[0]]
        elif mask.shape[0] > original_samples.shape[0]:
            mask = mask[:original_samples.shape[0]]

        # Apply noise based on blend mode
        if blend_mode == "replace":
            # High mask values = more noise, low mask values = less noise
            noised_samples = original_samples + random_noise * noise_strength
            result = mask * noised_samples + (1 - mask) * original_samples
        elif blend_mode == "add":
            # Additive noise scaled by mask
            result = original_samples + mask * random_noise * noise_strength
        elif blend_mode == "multiply":
            # Multiplicative noise scaled by mask
            result = original_samples * (1 + mask * random_noise * noise_strength)
        else:
            result = original_samples

        noise_latent["samples"] = result

        return (noise_latent, )

