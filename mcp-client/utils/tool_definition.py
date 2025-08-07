"""
    Utility functions to format MCP's defualt tool format to LLM/Inference provider specific format
"""

def google_definition(mcp_definition: dict) -> dict:
    """Converts a tool definition from MCP's default to Google's Gemini/Gemma models

    Args:
        mcp_definition: MCP's default tool definition.
    """

    gd = {
        "name": mcp_definition.get("name"),
        "description": mcp_definition.get("description"),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }

    input_schema = mcp_definition.get("input_schema", {})
    properties = input_schema.get("properties", {})

    for prop_name, prop_details in properties.items():
        gd["parameters"]["properties"][prop_name] = {
            "type": prop_details.get("type"),
            "description": prop_details.get("description"),
        }
        if "enum" in prop_details:
            gd["parameters"]["properties"][prop_name]["enum"] = prop_details.get("enum")
        # Assuming all properties in input_schema are required based on the example
        gd["parameters"]["required"].append(prop_name)

    return gd
    

def huggingface_definition(mcp_definition: dict) -> dict:
    """Converts a tool definition from MCP's default to HuggingfaceHub models

    Args:
        mcp_definition: MCP's default tool definition.
    """

    hfd = {
        "type": "function",
        "function": {
            "name": mcp_definition.get("name"),
            "description": mcp_definition.get("description"),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
    
    input_schema = mcp_definition.get("input_schema", {})
    properties = input_schema.get("properties", {})

    for prop_name, prop_details in properties.items():
        hfd["function"]["parameters"]["properties"][prop_name] = prop_details
        hfd["function"]["parameters"]["required"].append(prop_name)