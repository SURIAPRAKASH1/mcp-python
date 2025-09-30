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


def topk_tool_definition(tools: list[dict], topk_tools: list[dict]) -> list[dict]:
    """
    Filters the main tools list to include only those present in the topk_tools list.

    Args:
        tools: A list of dictionaries representing the full tool declarations.
        topk_tools: A list of dictionaries, each containing the 'name' and 'description'
                    of a top-k tool.

    Returns:
        A list of dictionaries containing the full tool declarations for the tools
        present in topk_tools.
    """
    # Create a dictionary of tools for efficient lookup by name
    tools_dict = {tool['name']: tool for tool in tools}

    # Filter the main tools list by iterating through topk_tools and looking up in tools_dict
    filtered_tools = [tools_dict[tool['name']] for tool in topk_tools if tool['name'] in tools_dict]

    return filtered_tools

