

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

