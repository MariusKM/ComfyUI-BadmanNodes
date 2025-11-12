

class BadmanIntUtil:
    def __init__(self):
        pass
    


    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Int1":  ("INT", {
                    "default": 0,
                    "min": 0}),
                "Int2":  ("INT", {
                    "default": 0,
                    "min": 0}),
                "math_function": (["add", "multiply", "sub", "divide"],),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "process_int"

    CATEGORY = "Badman"

    def process_int(self, Int1: int, Int2: int, math_function: str):
      
        processed_int = self.math_mode(Int1, Int2, math_function)
    
        return (processed_int,)

    def math_mode(self, Int1, Int2, mode):
        if mode == "add":
            return Int1+Int2
        elif mode == "multiply":
            return Int1 * Int2
        elif mode == "sub":
            return Int1 - Int2
        elif mode == "divide":
            return Int1/Int1
        else:
            raise ValueError(f"Unsupported math mode: {mode}")


class BadmanSelectFromList:
    """
    Select an item from any type of list by index.
    Supports positive and negative indices (Python-style).
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_list": ("INT", {
                    "tooltip": "List to select from"
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": -10000,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Index to select (supports negative indices)"
                }),
            },
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("item",)
    FUNCTION = "select_item"
    CATEGORY = "Badman"
    INPUT_IS_LIST = (True, False)
    OUTPUT_NODE = False
    
    def select_item(self, any_list, index):
        """
        Select an item from the list at the specified index.
        
        Args:
            any_list: Any list type (will be received as a list due to INPUT_IS_LIST)
            index: Integer index (supports negative indices) - may come as list or single value
        
        Returns:
            The item at the specified index
        """
        # Handle empty list
        if not any_list or len(any_list) == 0:
            raise ValueError("Cannot select from an empty list")
        
        # Unwrap double-wrapped lists (INPUT_IS_LIST wraps an already-wrapped list)
        # If we have [[81, 81, 49]], unwrap to [81, 81, 49]
        if len(any_list) == 1 and isinstance(any_list[0], list):
            any_list = any_list[0]
        
        # Extract index if it comes as a list
        if isinstance(index, list):
            index = index[0] if len(index) > 0 else 0
        
        # Handle index out of bounds
        try:
            item = any_list[index]
            return (item,)
        except IndexError:
            raise IndexError(f"Index {index} is out of bounds for list of length {len(any_list)}")

