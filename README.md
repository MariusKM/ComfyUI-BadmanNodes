# Badman Utility Nodes

A small collection of Utility and quality of Life Nodes developed to aid me in my workflows. Nothing special and nothing crazy, just extending certain functionalities of base nodes.


# Nodes

## Conditioning

**BadmanCLIPTextEncodeSDXLRegion** : SDXL conditioning Node intended for use with [CutOff](https://github.com/BlenderNeko/ComfyUI_Cutoff), sadly SDXL does not do too well with CutOff.

## Image

**ImageBlend (Badman)** : Extended Image Blend node with some extra blend functions.

**PalletteGenerator (Badman)** : Generates a grid of random colors, useful as a palette reference or for procedural color sampling.

**LABColorTransfer (Badman)** : Transfers the color statistics of a reference image onto a target image in LAB color space.

**Image Brightness Adjust (Badman)** : Simple brightness adjustment for images.

**Image Desaturate (Badman)** : Desaturates an image with adjustable strength.

**Mask Blur (Badman)** : Applies a gaussian blur to a mask.

**Dilate Erode Mask (Badman)** : Dilates or erodes a mask by a configurable amount.

## Strings

**String (Badman)** : Simple String Type node.

**Concat String (Badman)** : Simple String Concat Node with a new line option, ideal for combining Tokens from BLIP or CLIP interrogation.

**Print (Badman)** : Prints String input to console.

**Select String from List (Badman)** : Selects a specific String from a String List output and forwards this to the output. Can be used to target a specific String when loading prompts from files or from a multi image BLIP interrogator.

**Select from Any List (Badman)** : Generic version of the string selector that works on any list type, returning the item at the requested index.

**StringToInteger (Badman)** : Parses a string into an integer for use with numeric inputs.

**Wildcard Processor (Badman)** : Resolves wildcard tokens in prompt strings against wildcard text files.

**Prompt File Image Loader (Badman)** : Loads an image from a directory together with the matching prompt at the same index from a delimited prompt file. Supports a configurable delimiter and index wrapping, outputting image, mask, positive/negative prompt strings, and the resolved index.

## Numbers

**Int Math (Badman)** : Integer Math node with some basic Math functions.

**HexGenerator (Badman)** : Node that generates Hex Colors from linear RGB Values.

## IO

**IO Config (Badman)** : IO configurator that sets up paths dynamically to be stored in setter nodes.

## Latent / Noise

**Inject Latent Noise Masked (Badman)** : Injects noise into latent space with mask control. High mask values receive more noise, low values receive less. Supports multiple blend modes (replace, add, multiply) and mask inversion.

## Color / Matte

**Color Match (Badman)** : Corrects color drift in generated images by learning a transform from a reference pair (drifted → clean) and applying it to new images. Supports `reinhard` (LAB mean+std, safe across palettes) and `hist` (LAB per-channel histogram matching, stronger but palette-specific). Can save the learned transform as a preset and reload it later. Includes a live ~180 ms preview that updates when you drag `strength` or swap presets without re-queueing the workflow.

**Color Match Combine Presets (Badman)** : Merges multiple saved color-match presets into a single averaged preset — useful for building a "universal" correction from several character-specific pairs. 2–10 preset slots, all presets must share the same method.

**Chroma Clean (Badman)** : Refines a matte from BiRefNet / RMBG 2.0 / similar against a known-color backdrop. Punches out backdrop leaks (chroma fill-in) and despills edge halos by clamping the dominant background channel. Outputs despilled RGB, refined alpha, and the raw chroma-detection mask for tuning.

**Tiled Background Removal (Badman)** : Wrapper around BiRefNet / RMBG 2.0 that splits the input into overlapping tiles, runs inference per tile at native resolution, and reassembles a full-resolution mask with feathered seams. Recovers fine-edge detail (hair, crown tips, cavity outlines) that a single-pass net loses to its internal 1024² downsample. Supports an optional `safezone_mask` input to force-keep regions the matte net tends to drop (eyes, earrings, thin props). **Requires the [`comfyui-rmbg`](https://github.com/1038lab/ComfyUI-RMBG) pack** installed alongside this one; other nodes load regardless.

## WAN Video

**WAN Three Frame To Video (Badman)** : Generates video from 3 keyframes (start, middle, end) with proper masking for smooth transitions. Supports adjustable middle frame positioning, configurable frame blend width for smooth transitions, and CLIP vision output concatenation for enhanced conditioning.

**WAN Outpaint Frame Calculator (Badman)** : Computes the frame counts and padding needed to outpaint a WAN video clip to a target length.


## TODO

 - IO Config (Badman) : Refactor to allow for renaming specific I/O paths, currently fixed path names that I tend to use.
