import os, sys, time, signal, argparse
from typing import Optional, Dict, Any, List, Literal, get_args
from functools import partial
import inspect

# Configure logging
from logging_utils import get_logger
logger = get_logger(name = __name__, log_file= "host.log")

try:
    import asyncio 
    import gradio as gr 
    from utils.models import Chat
    from utils.format import Formatter
    from client import MCPClientManager

    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.error("%s", ImportError)
    sys.exit(1)
    
STREAMABLEHTTP_SERVER_URL = os.environ.get("STREAMABLEHTTP_SERVER_URL", "False")


# Mutuble container to Hold current app state
class AppState:
    mcp_manager = None 
    gradio_interface = None


def setup_signal_handling(state: AppState):
    """
    Set up signal handling in the main thread where signals are properly delivered.
    
    This is much simpler and more reliable than trying to handle signals
    in separate threads.
    """
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        
        # Stop the MCP manager and Gradio manager
        if state.mcp_manager:
            state.mcp_manager.stop()
        if state.gradio_interface and state.gradio_interface.is_running:
            state.gradio_interface.close()
        # Exit gracefully
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination
    
    # Windows-specific
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)


class GradioManager:
    """GradioManager to handle gradio related stuffs.

    Features:
        1. Encapsulated Gradio Functions and Components
        2. Connect to MCPClientManager via UI and Process MCP client with LLM
        3. Default Gracefull Error Handling
    """

    def __init__(self, state: AppState)-> None:  

        self.state = state
        if self.state.mcp_manager:
            # Partially created mcp methods for easy reusuability 
            self.list_tools = partial(self.state.mcp_manager.call_mcp_method, method = "list_tools", params = {}) 
            self.call_tool = partial(self.state.mcp_manager.call_mcp_method, method = "call_tool")
        else:
            self.list_tools = None
            self.call_tool = None

    def _connect_client(self, transport: str, server_arg: str = None):
        # Cleanup old client if it exists
        if self.state.mcp_manager:
            logger.info("Stopping Already Exits MCP Client ...")
            self.state.mcp_manager.stop()
            self.state.mcp_manager = None
        
        # Create new client manager instance
        self.state.mcp_manager = MCPClientManager() 

        if transport == "stdio":
            if not server_arg:
                raise ValueError("You must provide the path to the stdio server script.")
            self.state.mcp_manager.start_mcp_client(server_command= server_arg) 

        elif transport == "streamable-http":
            if not server_arg and STREAMABLEHTTP_SERVER_URL == "False" :
                raise ValueError("STREAMABLEHTTP_SERVER_URI env is not set. Use env OR use UI.")
            self.state.mcp_manager.start_mcp_client([], streamablehttp_url= STREAMABLEHTTP_SERVER_URL or server_arg[0])

        else:
            raise ValueError(f"Unknown transport type: {transport}")
        
        return f"Connected via {transport} transport."   
    
    def _handle_connect(self, transport: str, server_arg: str) -> None:
        """Connect MCP client with MCP server.
        
        Args:
            transport: Transport type client will connect server via.
            server_arg: Server args to connect to mcp server 
        """
        if isinstance(server_arg, str):
                server_arg = server_arg.strip()
                if not server_arg:
                    return "❌ Server command is empty."
                server_arg = server_arg.split(" ") # creates list of str
        else:
            raise ValueError(f"Invalid datatype for server_arg {server_arg}")
        
        try:

            status = self._connect_client(transport, server_arg)

            # update with new MCP Manager
            self.list_tools = partial(self.state.mcp_manager.call_mcp_method, method = "list_tools", params = {}) 
            self.call_tool = partial(self.state.mcp_manager.call_mcp_method, method = "call_tool")
            return status
        except Exception as e:
            return f"❌ Connection failed: {str(e)}"

    async def _handle_user_messages(self, message: str, history: List[Dict[str, Any]]):
        """Handle user input via LLM and communicate with MCP client.
        
        Args:
            messages: Current query from an user.
            history: All conversation between User & Assistant.
        """
        if not self.state.mcp_manager:
            yield "⚠️ No MCP server connection. Please connect first."
            return
        
        if not message.strip():
            yield "❌ message is empty. Ask something to starting the conversation !"
            return
        
        try:
            response = await asyncio.to_thread(self.list_tools, timeout = 15)
            if response['error']:
                yield f"❌ Error: {response['error']}"
                return

            # Defualt Structuring
            messages = Formatter.prepare_chat_messages(message, history)
            tools = Formatter.prepare_default_tool_definition(response["result"])

            if not tools:
                yield "✅ MCP server connected, but no tools available."
                return
            
            # Process query and tools via Custom implementation for Google's Models
            # llm_response = Chat("google").google(self.call_tool,
            #                                     tool_model="gemini-2.0-flash-lite",
            #                                     response_model= "gemma-3n-e4b-it", 
            #                                     messages= messages, 
            #                                     tools= tools
            #                                 )
            # return llm_response      
                    
            llm_obj = Chat("notebook").custom(
                call_tool= self.call_tool, 
                messages= messages, 
                tools = tools, 
            )

            # CASE A. async generator -> iterate and yield
            current_reasoning = ""
            accumulated_response = ""
            final_reasoning: Optional[gr.ChatMessage] = None
            table_mode = False
            table_buffer = ""

            if inspect.isasyncgen(llm_obj):
                async for item in llm_obj:
                    if isinstance(item, tuple) and len(item) == 2:
                        chunk, channel = item
                    else:
                        chunk, channel = item, None
                    chunk = chunk or ""

                    # Phase 1: Analysis -> LLM's CoT
                    if channel == "analysis":
                        current_reasoning += chunk
                        reasoning_msg = gr.ChatMessage(
                            role="assistant",
                            content= current_reasoning,
                            metadata={"title": "🧠 Reasoning Process", "status": "pending"}
                        )
                        yield reasoning_msg
                        time.sleep(0.5)

                    # Phase 2: Final channel
                    elif channel == "final":
                        # Table Response 
                        if not table_mode and Formatter.is_table_start(chunk):  
                            table_mode = True
                            table_buffer = chunk
                            
                            # Show thinking message while processing table
                            accumulated_response += "[📊 Processing table data...]\n\n"
                            response_msg = gr.ChatMessage(
                                    role="assistant",
                                    content=accumulated_response
                                )
                            
                        if not table_mode:
                            accumulated_response += chunk
                            response_msg = gr.ChatMessage(
                                            role="assistant",
                                            content=accumulated_response
                                        )
                        
                        if table_mode:
                            table_buffer += chunk
                            
                            if Formatter.is_table_end(table_buffer, chunk):
                                # Remove processing message and add formatted table
                                accumulated_response = accumulated_response.replace("[📊 Processing table data...]\n\n", "")
                                formatted_table = Formatter.format_table_content(table_buffer = table_buffer)
                                accumulated_response += formatted_table
                                
                                # Return to normal assistant state
                                response_msg = gr.ChatMessage(
                                    role="assistant", 
                                    content= accumulated_response
                                )
                                
                                table_mode = False
                                table_buffer = ""
                            else:
                                continue 

                        # Add a completion note to reasoning
                        if current_reasoning:
                            if final_reasoning is None:
                                current_reasoning += "\n✅ Analysis complete! Generating response..."
                                final_reasoning = gr.ChatMessage(
                                    role="assistant",
                                    content=current_reasoning,
                                    metadata={"title": "🧠 Reasoning Process", "status": "done"}
                                ) 

                            # Yield both the completed reasoning AND the growing response
                            yield [final_reasoning, response_msg]
                            time.sleep(0.4)
                        else:
                            yield response_msg
                            time.sleep(0.4)
                    # Phase 3: Unknown channel
                    else:
                        yield chunk 

            # CASE B. coroutine -> awaitable to get return result
            else:
                if inspect.iscoroutine(llm_obj):
                    res = await llm_obj
                else:
                    # CASE C. Just synchronous -> run in thread to avoid blocking eventloop
                    res = await asyncio.to_thread(llm_obj)

                # interpret result
                if isinstance(res, tuple) and len(res) == 2:
                    text, channel = res
                    final_text = text or f"channel {channel}"
                else:
                    final_text = str(res)
                yield final_text 

        except Exception as e:
            logger.error(f"Error in Gradio handler: {e}")
            yield f"❌ Unexpected error: {str(e)}"
            return

    def create_gradio_interface(self):
        """Create the Gradio interface with proper error handling."""
            
        # UI to choose connection type and connect
        with gr.Blocks() as interface:
            gr.Markdown("## MCP Client Interface")

            with gr.Row():
                transport_choice = gr.Dropdown(
                    ["stdio", "streamable-http"],
                    label="Transport Type",
                    value="stdio"
                )
                server_arg = gr.Textbox(
                    label="Server commands or URL",
                    placeholder="e.g, python ./server.py (if stdio)"
                )
                connect_btn = gr.Button("Connect")
                connect_status = gr.Markdown()

            # First Disable button and then run task, after task is done re-enable the button
            connect_btn.click(fn = self._disable_button, inputs= None, outputs= connect_btn).then(
                fn= self._handle_connect, 
                inputs= [transport_choice, server_arg], 
                outputs= connect_status
            ).then(
                fn= self._enable_button, 
                inputs= None, 
                outputs= connect_btn
            )

            gr.ChatInterface(
                    fn= self._handle_user_messages,
                    type="messages",
                    chatbot=gr.Chatbot(height=500, type= "messages")
            )

        return interface
    
    def _disable_button(self):
        "Disable the Button and shows connecting message"
        return gr.Button(interactive= False ,value= "Connecting ...") 

    def _enable_button(self):
        "Enable the Button and resets the text"
        return gr.Button(interactive= True, value= "Connect")


def main(server_command: Optional[List], streamablehttp_url = Optional[str]):
    """
    Main function with improved architecture.
    
    Key improvements:
    1. Signal handling stays in main thread
    2. Clear separation of concerns
    3. Robust error handling
    4. Simpler threading model
    """
    state = AppState()
    try:
        # Set up signal handling FIRST (in main thread)
        setup_signal_handling(state)
        
        # Create and start MCP manager if Only server commands are given
        if server_command or streamablehttp_url or STREAMABLEHTTP_SERVER_URL != "False":
            logger.info("Starting MCP client manager...")
            state.mcp_manager = MCPClientManager()
            if server_command:
                state.mcp_manager.start_mcp_client(server_command= server_command)
            elif streamablehttp_url or STREAMABLEHTTP_SERVER_URL:
                state.mcp_manager.start_mcp_client([], streamablehttp_url= streamablehttp_url or STREAMABLEHTTP_SERVER_URL)
        else:
            logger.info("You didn't start MCP client. Use gradio UI to start MCP client...")
        
        # Create Gradio interface 
        logger.info("Creating Gradio interface...")
        gradio_manager = GradioManager(state= state)
        state.gradio_interface = gradio_manager.create_gradio_interface()
        
        # Launch Gradio (this blocks the main thread)
        logger.info("Launching Gradio interface...")
        state.gradio_interface.launch(
            share= True, # don't use share if try to deploy in huggingface space
            debug=True
        )
        state.gradio_interface.queue() 
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        # Cleanup if Not
        if state.mcp_manager and not state.mcp_manager.is_mcp_gracefull_close:
            state.mcp_manager.stop()
        if state.gradio_interface and state.gradio_interface.is_running:
            state.gradio_interface.close()

        logger.info("Application finished")
        logger.info(f"==== Status ==== \nMCP:{"Closed" if state.mcp_manager and state.mcp_manager.is_mcp_gracefull_close else "Not Closed or May not Started."}\nGradio: {"Closed" if state.gradio_interface and not state.gradio_interface.is_running else "Not Closed or May not Started"}" )


if __name__ == "__main__": 

    parser = argparse.ArgumentParser() 
    parser.add_argument("--server_command", nargs="+", type= str, help= "server commands to connect to stdio server. Use space to sperate commands")
    parser.add_argument("--streamablehttp_url", type= str, help="URL for streamable-http server (eg., end with /mcp). Like gradio mcp server running in huggingface spaces or in local)") 

    args = parser.parse_args()
    main(args.server_command, args.streamablehttp_url)