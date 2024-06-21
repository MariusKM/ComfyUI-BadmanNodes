
class BadmanCLIPTextEncodeSDXLRegion:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 1024.0, "min": 0, "max": 4096}),
            "height": ("INT", {"default": 1024.0, "min": 0, "max": 4096}),
            "crop_w": ("INT", {"default": 0, "min": 0, "max": 4096}),
            "crop_h": ("INT", {"default": 0, "min": 0, "max": 4096}),
            "target_width": ("INT", {"default": 1024.0, "min": 0, "max": 4096}),
            "target_height": ("INT", {"default": 1024.0, "min": 0, "max": 4096}),
            "text_g": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ),
            "text_l": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ),
            }}
    RETURN_TYPES = ("CLIPREGION",)
    FUNCTION = "encode"

    CATEGORY = "Badman"


    def init_prompt(self, clip, text_g):
        tokens = clip.tokenize(text_g, return_word_ids=True)
        return ({
            "clip" : clip,
            "base_tokens" : tokens,
            "regions" : [],
            "targets" : [],
            "weights" : [],
        },)

    def encode(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l):
        # Tokenize the global text and store in the "g" key of tokens
        tokens_g = clip.tokenize(text_g, return_word_ids=True)
        tokens_l = clip.tokenize(text_l, return_word_ids=True)
        
        # Initialize the tokens dictionary with proper keys
        tokens = {
            "g": tokens_g.get("g", tokens_g),  # Fallback to tokens_g if "g" key is not present
            "l": tokens_l.get("l", tokens_l)   # Fallback to tokens_l if "l" key is not present
        }

        # Ensure the length of tokens["l"] matches the length of tokens["g"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            empty_l = empty.get("l", empty)  # Fallback to empty if "l" key is not present
            empty_g = empty.get("g", empty)  # Fallback to empty if "g" key is not present
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty_l
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty_g
        print(tokens)
        return ({
            "clip": clip,
            "base_tokens": tokens,
            "regions": [],
            "targets": [],
            "weights": [],
        },)


    """def encode(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l):
        tokens = clip.tokenize(text_g)
        tokens["l"] = clip.tokenize(text_l)["l"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        return ({
                "clip" : clip,
                "base_tokens" : tokens,
                "regions" : [],
                "targets" : [],
                "weights" : [],
            },)"""