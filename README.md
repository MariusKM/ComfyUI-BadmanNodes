# Badman Utiliity Nodes

A small collection of Utility and quality of Life Nodes developed to aid me in my workflows. Nothing special and nothing crazy, just extending certain functionalities of base nodes.


# Files

**BadmanCLIPTextEncodeSDXLRegion** : SDXL conditioning Node intended for use with [CutOff](https://github.com/BlenderNeko/ComfyUI_Cutoff),  sadly SDXL does not do too well with CutOff.

**ImageBlend(Badman)** : Extended Image Blend node with some extra blend functions.

**Int Math (Badman)** : Integer Math node with some basic Math functions

**HexGenerator(Badman)** : Node that generates Hex Colors from linear RGB Values

**String (Badman)** : Simple String Type node

**Concat String (Badman)** : Simple String Concat Node with a new line option, ideal for combining Tokens from BLIP or CLIP 
interrogation

**Print (Badman)** : Prints String input to console

**IO Config (Badman)** : IO configurator that sets up paths dynamically to be stored in setter nodes

**Select String from List (Badman)**: Selects a specific String from a String List output and forwards this to the output. Can be used to target a specific String when loading prompts from files or from a multi image BLIP interrogator.


## TODO

 - IO Config (Badman) : Refactor to allow for renaming specific I/O paths, currently fixed path names that I tend to use. 
