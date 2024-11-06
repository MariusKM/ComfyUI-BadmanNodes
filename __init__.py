from .BadmanStrings import *  
from .BadmanImage import * 
from .BadmanNumbers import * 
from .BadmanConditioning import *
from .BadmanColorTransfer import *
from .BadmanWildCardProcessor import *


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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Badman_Blend": "ImageBlend(Badman)",
    "Badman_PalletteGenerator": "PalletteGenerator(Badman)",
    "Badman_HexGenerator": "HexGenerator(Badman)",
    "Badman_ColorTransferLab" : "LABColorTransfer(Badman)",
    "Badman_String": "String (Badman)",
    "Badman_Concat_String": "Concat String (Badman)",
    "Badman_Print": "Print (Badman)",
    "Badman_IO": "IO Config (Badman)",
    "BadmanIntUtil": "Int Math (Badman)",
    "BadmanStringSelect": "Select String from List (Badman)",
    "BadmanBrightness" : "Image Brightness Adjust (Badman)",
    "BadmanWildCardProcessor" : "Wildcard Processor (Badman)",
    "BadmanDesaturate" : "Image Desaturate (Badman)",
    "BadmanMaskBlur" : "Mask Blur (Badman)",
    "BadmanDilateErodeMask" : "Dilate Erode Mask (Badman)",
    "BadmanStringToInteger" : "StringToInteger (Badman)",
}
