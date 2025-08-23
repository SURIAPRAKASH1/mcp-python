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
    """Converts a tool definition from MCP's default to compatible with HuggingfaceHub models

    Args:
        mcp_definition: MCP's default tool definition.
    """

    hug_def = {
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
        hug_def["function"]["parameters"]["properties"][prop_name] = prop_details
        hug_def["function"]["parameters"]["required"].append(prop_name)

    return hug_def

def custom_definition(mcp_definition: dict)-> dict:
    """Converts a tool definition from MCP's default to compatible with Custom model running on notebook

    Args:
        mcp_definition: MCP's default tool definition.
    """
    
    cus_def = {
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

    # Handle enum
    prop_details_enum = {
        "type": [],
        "description": ""
    }
    
    input_schema = mcp_definition.get("input_schema", {})
    properties = input_schema.get("properties", {})

    for prop_name, prop_details in properties.items():
        # Argument is enum -> changing type to hold enum's 
        if prop_details["enum"]:
            prop_details_enum["type"] = prop_details["enum"]
            prop_details_enum["description"] = prop_details["description"]    
            prop_details = prop_details_enum

        cus_def["function"]["parameters"]["properties"][prop_name] = prop_details
        cus_def["function"]["parameters"]["required"].append(prop_name)

    return cus_def
