import asyncio
import httpx
from contextlib import AsyncExitStack
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp import Tool
import gradio as gr

from typing import List, Optional, Any, Union
import sys, json, os
from dotenv import load_dotenv
load_dotenv() 

GEMMA3N_URI = os.environ.get("GEMMA3N_URI", "False")
STREAMABLEHTTP_URI = os.environ.get("STREAMABLEHTTP_URI", "False")

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

    async def connect_to_streamablehttp_server(self, streamablehttp_uri: str)-> None:
        """Connect to streamablehttp MCP server

        Args:
            streamablehttp_uri: URI to where streamablehttp server is running (/mcp)
        """
        streamablehttp_transport = await self.exit_stack.enter_async_context(streamablehttp_client(url = streamablehttp_uri)) 
        read, write, _ = streamablehttp_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write)) 

        await self.session.initialize()

        # List available tools
        response =  await self.session.list_tools() 
        self.tools: List[Tool] = response.tools 
        print("\nConnected to streamablehttp server with tools: ", [tool.name for tool in self.tools])
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

        headers = {"user-agent": "mcp-client/0.0.1","content-type": "application/json"}

        async with httpx.AsyncClient(timeout= 10.0) as client:
            try:
                to_llm_context = {"messages": messages, "tools": available_tools}
                llm_response = await client.post(url = GEMMA3N_URI, headers= headers, json= to_llm_context)
                llm_response.raise_for_status()            
            except httpx.HTTPStatusError as exc:
                return f"Error Response {exc.response.status_code} while requesting {exc.request.url}"
            except httpx.RequestError as exc:
                return f"Error while requesting {exc.request.url}"
            except Exception as exc:
                return f"Unexpected Error when Accesssing LLM API: \n {exc}"

        final_text = []
        assistant_message_context = []
        response = llm_response.get('message')

        if response['content']:                       
            final_text.append(response['content'])
            assistant_message_context.append(response['content'])
        elif response['tool_calls']:

            for tool_call in response['tool_calls']:
                function_name, function_args = tool_call['function'].get('name'), tool_call['function'].get("arguments")

                # Execute tool call
                result = self.session.call_tool(function_name, function_args)
                print(f"Calling tool {function_name} with arguments {function_args}")
                
                messages.append({
                    'role': "assistant",
                    'content': tool_call['function']
                
                }) 
                # add tool result to LLM's context
                messages.append({
                    "role": "tool",
                    "content": result.content
                })

                # again invoke LLM 
                async with httpx.AsyncClient(timeout= 10.0) as client:
                    try:
                        to_llm_context = {"messages": messages, "tools": available_tools}
                        llm_response = await client.post(url = GEMMA3N_URI, headers= headers, json= to_llm_context)
                        llm_response.raise_for_status()            
                    except httpx.HTTPStatusError as exc:
                        return f"Error Response {exc.response.status_code} while requesting {exc.request.url}"
                    except httpx.RequestError as exc:
                        return f"Error whild requesting {exc.request.url}"
                    except Exception as exc:
                        return f"Unexpected Error when Accesssing LLM API: \n {exc}"
                
                final_text.append(llm_response['message'].get("content"))

            return "\n".join(text for text in final_text)

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

async def main():
    
    is_stdio_server = True
    is_streamablehttp_server = True 

    if len(sys.argv) < 2:
        is_stdio_server = False
    if STREAMABLEHTTP_URI == "False":
        is_streamablehttp_server = False

    if not (is_stdio_server or is_streamablehttp_server):
        print("Usage: python client.py <path to your server script>")
        print("OR")
        print("Set env variable STREAMABLE_HTTP_URI = <uri to your server>")
        sys.exit(1)

    client = MCPClient()

    try:
        if is_stdio_server:
            await client.connect_to_stdio_server(sys.argv[1]) 
        elif is_streamablehttp_server:
            await client.connect_to_streamablehttp_server(STREAMABLEHTTP_URI)
    
        gr.ChatInterface(
            fn= client.process_query, 
            type = "messages", 
            chatbot= gr.Chatbot(height= 500),
        ).launch(
            debug = True, 
        )        
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

