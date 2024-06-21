import os

class BadmanPrint:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("STRING", {"default": ""})},
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "log_input"
    CATEGORY = "Badman"

    def log_input(self, value):
        print(value,)
        return {}
    
class ConcatString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
             "required": {
                "string_field_0": ("STRING", {
                    "multiline": False,
                    "default": "Hello"
                }),
                "string_field_1": ("STRING", {
                    "multiline": False,
                    "default": "World"
                }),
                "newline": ("BOOLEAN",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Badman"

    def execute(self, string_field_0, string_field_1,newline):
        if newline:
            return (string_field_0 +"\n"+string_field_1,)
        else:
            return (string_field_0 + string_field_1,)
class SelectString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
             "required": {
                "string_List": ("STRING", {
                    
                }),
                "Index": ("INT", {
                    "default": 0,
                    "min": 0}),
            },
        }
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Badman"

    def execute(self,string_List ,Index ):
       
        return (string_List[Index[0]],)

class BadmanString:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Badman"

    def execute(self, value):
        return (value,)
    
class BadmanIOConfigurator:
    # Get the directory of the current file
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # Get the parent directory
    Output_directory = os.path.dirname(os.path.dirname(current_file_directory))+"\\output\\"
 
    
    @classmethod
    def INPUT_TYPES(s):
         return {
            "required": {
                "BasePath": ("STRING", {"default": s.Output_directory}),
                "ShotName": ("STRING", {"default": ""}),
                },
        }
    
    RETURN_TYPES = ("STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING",)
    RETURN_NAMES = ("ImageOutputPath","VideoOutputPath","ImageUpresOutputPath","VideoUpresOutputPath","ImageInputPath","DepthInputPath","PoseInputPath","BodyMaskInputPath","FaceMaskInputPath",)
    FUNCTION = "execute"
    CATEGORY = "Badman"


    def execute(self, BasePath,ShotName):
        suffixOut = BasePath+ShotName
        imagesOutPath = suffixOut+"\Images\image"
        videoOutPath = suffixOut+"\Video"
        imagesUpresOutPath = suffixOut+"\ImagesUpres\image"
        videoUpresOutPath = suffixOut+ "\VideoUpres"
        framesInPath = BasePath+"ExtractedFrames\\"+ShotName+"\\"
        depthInPath = BasePath+"ExtractedFrames\\"+ShotName+"\\"
        poseInPath = BasePath+"ExtractedPose\\"+ShotName+"\\"
        BodyMaskInPath = BasePath+"Masks\\"+ShotName+"\\Body\\"
        FaceMaskInPath = BasePath+"Masks\\"+ShotName+"\\Face\\"
       
        return (imagesOutPath,videoOutPath,imagesUpresOutPath,videoUpresOutPath,framesInPath,depthInPath,poseInPath,BodyMaskInPath,FaceMaskInPath)
       


