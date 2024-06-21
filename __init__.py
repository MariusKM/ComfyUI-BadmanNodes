from .BadmanStrings import *  
from .BadmanImage import * 
from .BadmanNumbers import * 
from .BadmanConditioning import *


NODE_CLASS_MAPPINGS = {
    "Badman_Blend": Blend,
    "Badman_HexGenerator": HexGenerator,
    "Badman_String": BadmanString,
    "Badman_Concat_String": ConcatString,
    "Badman_Print": BadmanPrint,
    "BadmanIO" : BadmanIOConfigurator,
    "BadmanIntUtil" : BadmanIntUtil,
    "BadmanCLIPTextEncodeSDXLRegion" : BadmanCLIPTextEncodeSDXLRegion,
    "BadmanStringSelect" : SelectString,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Badman_Blend": "ImageBlend(Badman)",
    "Badman_HexGenerator": "HexGenerator(Badman)",
    "Badman_String": "String (Badman)",
    "Badman_Concat_String": "Concat String (Badman)",
    "Badman_Print": "Print (Badman)",
    "Badman_IO": "IO Config (Badman)",
    "BadmanIntUtil": "Int Math (Badman)",
    "BadmanStringSelect": "Select String from List (Badman)",
    "BadmanCLIPTextEncodeSDXLRegion" : "CLIPTextEncodeSDXLRegion (Badman)",
}
