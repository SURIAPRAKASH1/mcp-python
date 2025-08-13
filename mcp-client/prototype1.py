from typing import List, Optional, Any, Union
import sys, json, os
from dotenv import load_dotenv
load_dotenv() 

from utils.models import Chat

try:
    import asyncio
    import httpx
    from contextlib import AsyncExitStack
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.sse import sse_client
    from mcp.types import Tool, Prompt
    import gradio as gr
    import atexit
except ImportError:
    raise ImportError



GEMMA3N_URI = os.environ.get("GEMMA3N_URI", "False")
STREAMABLEHTTP_SERVER_URI = os.environ.get("STREAMABLEHTTP_SERVER_URI", "False")
SSE_SERVER_URI = os.environ.get("SSE_SERVER_URI", "False")

class MCPClient():
    """ MCPClient: Client that will connect to MCP server and performers MCP Protocals"""

    def __init__(self) -> None:
        
        self.session: Optional[ClientSession] = None 
        self.exit_stack = AsyncExitStack() 
    
    async def connect_to_stdio_server(self, server_script_path: str) -> None:
        """Connect to an stdio MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None, 
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        self.tools: List[Tool]  = response.tools
        print("\nConnected to stdio server with tools:", [tool.name for tool in self.tools])
        print(" ")

    async def connect_to_streamablehttp_server(self, streamablehttp_server_uri: str)-> None:
        """Connect to an streamablehttp MCP server

        Args:
            streamablehttp_server_uri: URI to where streamablehttp server is running (e.g, /mcp)
        """
        streamablehttp_transport = await self.exit_stack.enter_async_context(streamablehttp_client(url = streamablehttp_server_uri)) 
        read, write, _ = streamablehttp_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write)) 

        await self.session.initialize()

        # List available tools
        response =  await self.session.list_tools() 
        self.tools: List[Tool] = response.tools 
        print("\nConnected to streamablehttp server with tools: ", [tool.name for tool in self.tools])
        print(" ")
    
    # gradio>=5.41.0 advanced to streamable-http so no need to use sse.
    async def connect_to_sse_server(self, sse_server_uri: str) -> None:
        """Connect to an sse MCP server. This implementation is to only handle gradio. Cause when i was implementing gradio didn't advance to streamable-http. 

        Args:
            sse_server_uri: URI to where see server is running (e.g, mcp/see)
        """
        sse_transport = await self.exit_stack.enter_async_context(sse_client(url = sse_server_uri)) 
        read, write = sse_transport 
        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write)) 

        await self.session.initialize() 

        # List available tools 
        tool_response = await self.session.list_tools() 
        # List available prompts 
        prompt_response = await self.session.list_prompts() 

        self.tools: List[Tool] = tool_response.tools 
        self.prompts: List[Prompt] = prompt_response.prompts
        print(f"\nConnected to sse server with \ntools: {[tool.name for tool in self.tools]} \nprompts: {[prompt.name for prompt in self.prompts]}")
        print(" ")
        

    async def process_query(self, message: str, history: list[dict[str, Any]]) -> str:
        """Process a query using LLM and available tools

        Args:
            messages: Current query from an user
            history: All conversation between User & Assistant
        """

        messages = self.prepare_chat_messages(message, history)

        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in self.tools]

        # response = await Chat(plateform= "google").google(self.session, 
        #                                                 tool_model= "gemini-2.0-flash-lite", 
        #                                                 response_model= None, 
        #                                                 messages= messages, 
        #                                                 tools= available_tools)  

        response = await self.session.call_tool(name = "analyze_file_changes")

        return f"{response.structuredContent}"

    def prepare_chat_messages(self, message: str, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prepares User and Assistant chat history in a structure way"""

        # History of user - assistant chat
        messages = []
        if history:
            # go through all conversation extract user & assistant 
            for hist in history:
                if hist.get("role") == "user":
                    messages.append({"role": "user", "content": hist.get("content")})
                elif hist.get("role") == "assistant":
                    messages.append({"role": "assistant", "content": hist.get("content")})

        # finaly append current user query to messages
        messages.append({"role": "user", "content": message})
        return messages

    async def cleanup(self):
        """Destroy session"""
        await self.exit_stack.aclose()

"""Starting MCP client before lauching gradio web server. This approch let gradio process on already exited MCP Client 🤝"""


# Global client reference so Gradio can use it
mcp_client: MCPClient | None = None

async def connect_client(transport: str, server_arg: str = None):
    global mcp_client

    # Cleanup old client if it exists
    if mcp_client:
        await mcp_client.cleanup()
        mcp_client = None

    client = MCPClient()

    if transport == "stdio":
        if not server_arg:
            raise ValueError("You must provide the path to the stdio server script.")
        await client.connect_to_stdio_server(server_arg)

    elif transport == "streamable-http":
        if STREAMABLEHTTP_SERVER_URI == "False":
            raise ValueError("STREAMABLEHTTP_SERVER_URI is not set.")
        await client.connect_to_streamablehttp_server(STREAMABLEHTTP_SERVER_URI)

    elif transport == "sse":
        if SSE_SERVER_URI == "False":
            raise ValueError("SSE_SERVER_URI is not set.")
        await client.connect_to_sse_server(SSE_SERVER_URI)

    else:
        raise ValueError(f"Unknown transport type: {transport}")

    mcp_client = client
    return f"Connected via {transport} transport."


async def process_query(message: str, history: List[dict[str, Any]]) -> str:
    if not mcp_client:
        return "⚠️ No MCP server connection. Please connect first."
    return await mcp_client.process_query(message, history)


def cleanup_on_exit():
    """Ensure cleanup when Gradio or the process stops."""
    loop = asyncio.get_event_loop()
    if mcp_client:
        loop.run_until_complete(mcp_client.cleanup())


# Register cleanup for process exit
atexit.register(cleanup_on_exit)


# UI to choose connection type and connect
with gr.Blocks() as demo:
    gr.Markdown("## MCP Client Interface")

    with gr.Row():
        transport_choice = gr.Dropdown(
            ["stdio", "streamable-http", "sse"],
            label="Transport Type",
            value="stdio"
        )
        stdio_path = gr.Textbox(
            label="Path to Stdio Server Script (if using stdio)",
            placeholder="example: ./server.py"
        )
        connect_btn = gr.Button("Connect")
        connect_status = gr.Markdown()

    chat = gr.ChatInterface(
        fn=process_query,
        type="messages",
        chatbot=gr.Chatbot(height=500, type= "messages")
    )

    async def handle_connect(transport, path):
        try:
            status = await connect_client(transport, path.strip() or None)
            return status
        except Exception as e:
            return f"❌ Connection failed: {str(e)}"

    # Have to connect to MCP server before chatting
    connect_btn.click(
        handle_connect,
        inputs=[transport_choice, stdio_path],
        outputs=connect_status
    )


if __name__ == "__main__":
    demo.launch(
        share= True, 
        debug= True
    )









