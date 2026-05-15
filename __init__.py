from .BadmanStrings import *  
from .BadmanImage import * 
from .BadmanNumbers import * 
from .BadmanConditioning import *
from .BadmanColorTransfer import *
from .BadmanWildCardProcessor import *
from .BadmanLatentNoiseMask import *
from .BadmanWanNodes import *
from .BadmanWanOutpaintNodes import *
from .BadmanPromptFileLoader import BadmanPromptFileImageLoader
from .BadmanColorMatch import ColorMatchNode, ColorMatchCombinePresets
from .BadmanChromaClean import ChromaCleanNode


NODE_CLASS_MAPPINGS = {
    "Badman_Blend": Blend,
    "Badman_PalletteGenerator": RandomColorImageGrid,
    "Badman_HexGenerator": HexGenerator,
    "Badman_ColorTransferLab": LabColorTransferNode,
    "Badman_String": BadmanString,
    "Badman_Concat_String": ConcatString,
    "Badman_Print": BadmanPrint,
    "BadmanIO" : BadmanIOConfigurator,
    "BadmanIntUtil" : BadmanIntUtil,
    "BadmanCLIPTextEncodeSDXLRegion" : BadmanCLIPTextEncodeSDXLRegion,
    "BadmanStringSelect" : SelectString,
    "BadmanBrightness" : Brightness,
    "BadmanWildCardProcessor" : BadmanWildCardProcessor,
    "BadmanDesaturate" : ImageDesaturate,
    "BadmanMaskBlur" : MaskBlur,
    "BadmanDilateErodeMask" : DilateErodeMask,
    "BadmanStringToInteger" : StringToInteger,
    "BadmanInjectLatentNoiseMasked" : InjectLatentNoiseMasked,
    "BadmanWanThreeFrameToVideo" : WanThreeFrameToVideo,
    "BadmanWanOutpaintFrameCalculator" : WanOutpaintFrameCalculator,
    "BadmanSelectFromList" : BadmanSelectFromList,
    "BadmanPromptFileImageLoader" : BadmanPromptFileImageLoader,
    "BadmanColorMatch" : ColorMatchNode,
    "BadmanColorMatchCombine" : ColorMatchCombinePresets,
    "BadmanChromaClean" : ChromaCleanNode,
}

# Tiled RMBG depends on the comfyui-rmbg pack. Register it only if that pack
# is installed — other nodes in this pack should still load regardless.
try:
    from .BadmanTiledRMBG import TiledRMBGNode
    NODE_CLASS_MAPPINGS["BadmanTiledRMBG"] = TiledRMBGNode
    print("[Badman] Tiled Background Removal (Badman) registered")
except Exception as _e:
    import traceback
    print(f"[Badman] Tiled Background Removal NOT registered: {type(_e).__name__}: {_e}")
    traceback.print_exc()

NODE_DISPLAY_NAME_MAPPINGS = {
    "Badman_Blend": "ImageBlend(Badman)",
    "Badman_PalletteGenerator": "PalletteGenerator(Badman)",
    "Badman_HexGenerator": "HexGenerator(Badman)",
    "Badman_ColorTransferLab" : "LABColorTransfer(Badman)",
    "Badman_String": "String (Badman)",
    "Badman_Concat_String": "Concat String (Badman)",
    "Badman_Print": "Print (Badman)",
    "BadmanIO": "IO Config (Badman)",
    "BadmanIntUtil": "Int Math (Badman)",
    "BadmanStringSelect": "Select String from List (Badman)",
    "BadmanBrightness" : "Image Brightness Adjust (Badman)",
    "BadmanWildCardProcessor" : "Wildcard Processor (Badman)",
    "BadmanDesaturate" : "Image Desaturate (Badman)",
    "BadmanMaskBlur" : "Mask Blur (Badman)",
    "BadmanDilateErodeMask" : "Dilate Erode Mask (Badman)",
    "BadmanStringToInteger" : "StringToInteger (Badman)",
    "BadmanInjectLatentNoiseMasked" : "Inject Latent Noise Masked (Badman)",
    "BadmanWanThreeFrameToVideo" : "WAN Three Frame To Video (Badman)",
    "BadmanWanOutpaintFrameCalculator" : "WAN Outpaint Frame Calculator (Badman)",
    "BadmanSelectFromList" : "Select from Any List (Badman)",
    "BadmanPromptFileImageLoader" : "Prompt File Image Loader (Badman)",
    "BadmanColorMatch" : "Color Match (Badman)",
    "BadmanColorMatchCombine" : "Color Match Combine Presets (Badman)",
    "BadmanChromaClean" : "Chroma Clean (Badman)",
    "BadmanTiledRMBG" : "Tiled Background Removal (Badman)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
