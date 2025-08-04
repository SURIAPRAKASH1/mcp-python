import inspect 
from primitives import tools 

def get_tool_functions():
    "Only get async function and function is tool from tools.py"

    return {
        name: func
        for name, func in inspect.getmembers(tools, inspect.iscoroutinefunction)
        if not name.startswith("_") 
    }
