import asyncio
import threading
import signal
import sys
import logging
import weakref
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import gradio as gr
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging to help debug the coordination
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCPGradioManager:
    """
    A comprehensive manager that coordinates MCP client and Gradio lifecycle.
    Think of this as the conductor orchestrating both the client and web interface.
    """
    
    def __init__(self):
        self.client_session: Optional[ClientSession] = None
        self.server_params: Optional[StdioServerParameters] = None
        self.gradio_app: Optional[gr.Interface] = None
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.loop_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        self.client_ready = threading.Event()
        
        # Use weak references to avoid circular references that prevent cleanup
        self._cleanup_callbacks = weakref.WeakSet()
        
    async def initialize_client(self, args: list, server_env: Optional[Dict[str, str]] = None):
        """
        Initialize the MCP client connection.
        This is like tuning the first instrument before the performance begins.
        """
        try:
            logger.info("Initializing MCP client...")
            
            # Create server parameters for stdio transport
            self.server_params = StdioServerParameters(
                command= "python",
                args = args,
                env=server_env or {}
            )
            
            # Establish the client session
            # The @asynccontextmanager ensures proper cleanup even if something goes wrong
            @asynccontextmanager
            async def create_session():
                async with stdio_client(self.server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        yield session
            
            # Store the context manager for later use
            self._session_manager = create_session()
            self.client_session = await self._session_manager.__aenter__()
            
            # Initialize the session (this is like the handshake between client and server)
            await self.client_session.initialize()
            
            logger.info("MCP client initialized successfully")
            self.client_ready.set()  # Signal that client is ready
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            self.client_ready.set()  # Set anyway to avoid blocking
            raise
    
    async def cleanup_client(self):
        """
        Properly clean up the MCP client connection.
        This is like carefully putting away the instruments after the performance.
        """
        if self.client_session and hasattr(self, '_session_manager'):
            try:
                logger.info("Cleaning up MCP client...")
                await self._session_manager.__aexit__(None, None, None)
                logger.info("MCP client cleanup completed")
            except Exception as e:
                logger.error(f"Error during MCP client cleanup: {e}")
            finally:
                self.client_session = None
    
    def start_event_loop(self):
        """
        Start the event loop in a separate thread.
        Think of this as starting the metronome that keeps everything in time.
        """
        def run_loop():
            # Create a new event loop for this thread
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            
            try:
                logger.info("Starting event loop...")
                self.event_loop.run_forever()
            except Exception as e:
                logger.error(f"Event loop error: {e}")
            finally:
                logger.info("Event loop stopped")
        
        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()
        
        # Wait a moment for the loop to be ready
        import time
        time.sleep(0.1)
    
    def stop_event_loop(self):
        """
        Stop the event loop gracefully.
        This is like gradually slowing down the metronome to a stop.
        """
        if self.event_loop and self.event_loop.is_running():
            logger.info("Stopping event loop...")
            self.event_loop.call_soon_threadsafe(self.event_loop.stop)
            
            # Wait for the loop thread to finish
            if self.loop_thread and self.loop_thread.is_alive():
                self.loop_thread.join(timeout=5.0)
                if self.loop_thread.is_alive():
                    logger.warning("Event loop thread did not stop gracefully")
    
    async def call_mcp_method(self, method: str, params: Dict[str, Any] = None) -> Any:
        """
        Call an MCP method safely.
        This is like passing a message between the musicians during the performance.
        """
        if not self.client_session:
            raise RuntimeError("MCP client not initialized")
        
        try:
            # Example of calling a tools/list method
            if method == "list_tools":
                result = await self.client_session.list_tools()
                return [tool.name for tool in result.tools]
            
            # Add other method handlers as needed
            return f"Method {method} called successfully"
            
        except Exception as e:
            logger.error(f"Error calling MCP method {method}: {e}")
            raise
    
    def schedule_async_task(self, coro):
        """
        Schedule an async task from a sync context (like Gradio callbacks).
        This is like sending a request from the audience to the performers.
        """
        if not self.event_loop:
            raise RuntimeError("Event loop not running")
        
        # Use asyncio.run_coroutine_threadsafe to bridge sync/async contexts
        future = asyncio.run_coroutine_threadsafe(coro, self.event_loop)
        return future.result(timeout=30.0)  # Wait up to 30 seconds
    
    def create_gradio_interface(self):
        """
        Create the Gradio interface that interacts with MCP client.
        This is like setting up the stage for the audience to interact with the performance.
        """
        def handle_user_input(message, history):
            """Handle user input and call MCP methods"""
            try:
                # Wait for client to be ready
                if not self.client_ready.wait(timeout=10.0):
                    return "MCP client not ready"
                
                # Example: List available tools
                result = self.schedule_async_task(
                    self.call_mcp_method("list_tools")
                )
                return f"Available tools: {result}"
                
            except Exception as e:
                logger.error(f"Error handling user input: {e}")
                return f"Error: {str(e)}"
        
        # Create the Gradio interface
        self.gradio_app = gr.ChatInterface(
            fn=handle_user_input,
            type = "messages", 
            chatbot= gr.Chatbot(type = "messages"),
            title="MCP Client Interface",
            description="Interact with MCP tools through this interface"
        )
        
        return self.gradio_app
    
    def setup_signal_handlers(self):
        """
        Set up signal handlers for graceful shutdown.
        This is like teaching the conductor how to respond to the audience leaving.
        """
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_event.set()
            self.graceful_shutdown()
            sys.exit(0)
        
        # Handle common termination signals
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination request
        
        # On Windows, also handle SIGBREAK
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def graceful_shutdown(self):
        """
        Perform graceful shutdown of all components.
        This is the carefully choreographed end of the performance.
        """
        logger.info("Starting graceful shutdown...")
        
        try:
            # Step 1: Close Gradio first (stops accepting new requests)
            if self.gradio_app and hasattr(self.gradio_app, 'close'):
                logger.info("Closing Gradio interface...")
                self.gradio_app.close()
            
            # Step 2: Clean up MCP client (close connection to server)
            if self.event_loop and self.client_session:
                logger.info("Cleaning up MCP client...")
                future = asyncio.run_coroutine_threadsafe(
                    self.cleanup_client(), 
                    self.event_loop
                )
                future.result(timeout=10.0)
            
            # Step 3: Stop the event loop
            self.stop_event_loop()
            
            logger.info("Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
    
    async def run_async(self, args: list, gradio_kwargs: Dict[str, Any] = None):
        """
        Main async method to run everything together.
        This is the complete performance from start to finish.
        """
        gradio_kwargs = gradio_kwargs or {}
        
        try:
            # Initialize the MCP client first
            await self.initialize_client(args)
            
            # Create and launch Gradio interface
            interface = self.create_gradio_interface()
            
            # Set default Gradio parameters
            launch_kwargs = {
                'server_port': 7860,
                'server_name': '0.0.0.0',
                'share': False,
                **gradio_kwargs
            }
            
            logger.info("Launching Gradio interface...")
            interface.launch(**launch_kwargs)
            
        except Exception as e:
            logger.error(f"Error in main execution: {e}")
            raise
    
    def run(self, args: list, gradio_kwargs: Dict[str, Any] = None):
        """
        Main synchronous method to run everything.
        This sets up the entire performance and handles the coordination.
        """
        try:
            # Set up signal handlers for graceful shutdown
            self.setup_signal_handlers()
            
            # Start the event loop in a separate thread
            self.start_event_loop()
            
            # Wait for event loop to be ready
            import time
            time.sleep(0.5)
            
            # Schedule the main async task
            logger.info("Starting MCP-Gradio integration...")
            future = asyncio.run_coroutine_threadsafe(
                self.run_async(args, gradio_kwargs),
                self.event_loop
            )
            
            # Wait for completion or interruption
            try:
                future.result()
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.shutdown_event.set()
            
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
        
        finally:
            # Ensure cleanup always happens
            if not self.shutdown_event.is_set():
                self.graceful_shutdown()


def main():
    """
    Example usage of the MCP-Gradio integration.
    This demonstrates how to use the manager in practice.
    """
    # Create the manager instance
    manager = MCPGradioManager()
    
    # Define your MCP server command
    # Replace this with your actual MCP server command
    args = []
    
    # Optional Gradio configuration
    gradio_config = {
        'server_port': 7860,
        'share': False,
        'debug': True
    }
    
    try:
        # Run the integrated system
        manager.run(args, gradio_config)
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        logger.info("Application finished")


if __name__ == "__main__":
    main()