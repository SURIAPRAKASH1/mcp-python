import gradio as gr 
from primitives import tools 
from registry import get_tool_functions 

# Auto register gradio components 
with gr.Blocks() as demo:
    for name, func in get_tool_functions().items(): 
        component = tools.TOOL_COMPONENTS.get(name)

        if component:
            if component["is_gradio_api"]:
                gr.Markdown(
                    """
                    This tool is MCP-only, so it doesn't have UI. Have to access programmatically. 
                    """
                    )
                gr.api(
                    fn = func, 
                    api_name= name, 
                )
            else:
                gr.Interface(
                    fn = func, 
                    inputs= component["inputs"], 
                    outputs = component["outputs"], 
                    title= name
                )

if __name__ == "__main__":
    demo.launch(
        share= True, 
        mcp_server = True
    )
