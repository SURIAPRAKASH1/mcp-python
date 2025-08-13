import asyncio
import threading
import signal
import sys
import logging
import weakref
import datetime
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
        
    async def initialize_client(self, server_command: list, server_env: Optional[Dict[str, str]] = None):
        """
        Initialize the MCP client connection.
        This is like tuning the first instrument before the performance begins.
        """
        try:
            logger.info("Initializing MCP client...")
            
            # Create server parameters for stdio transport
            self.server_params = StdioServerParameters(
                command= server_command[0],
                args=[server_command[1]],
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
        Call an MCP method safely with detailed error reporting.
        This function is like a careful translator ensuring messages are properly delivered.
        """
        import traceback
        
        # Pre-flight checks with detailed logging
        if not self.client_session:
            error_msg = "MCP client session is None - connection was never established or was lost"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            logger.info(f"Attempting to call MCP method: {method} with params: {params}")
            
            # Check if the session is still alive by testing basic functionality
            # This is like checking if the phone line is still connected before making a call
            try:
                # Try to list tools as a connectivity test
                if method == "list_tools":
                    logger.info("Calling list_tools method...")
                    result = await self.client_session.list_tools()
                    tool_names = [tool.name for tool in result.tools] if result.tools else []
                    logger.info(f"Successfully retrieved {len(tool_names)} tools: {tool_names}")
                    return tool_names
                
                elif method == "call_tool":
                    # Handle tool calling with proper parameter validation
                    if not params or 'tool_name' not in params:
                        raise ValueError("Tool name is required for call_tool method")
                    
                    tool_name = params['tool_name']
                    tool_args = params.get('arguments', {})
                    
                    logger.info(f"Calling tool '{tool_name}' with arguments: {tool_args}")
                    result = await self.client_session.call_tool(tool_name, tool_args)
                    logger.info(f"Tool call completed successfully")
                    return result
                
                elif method == "list_resources":
                    logger.info("Calling list_resources method...")
                    result = await self.client_session.list_resources()
                    resource_list = [resource.uri for resource in result.resources] if result.resources else []
                    logger.info(f"Successfully retrieved {len(resource_list)} resources")
                    return resource_list
                
                else:
                    # For other methods, provide a clear error message
                    available_methods = ["list_tools", "call_tool", "list_resources"]
                    error_msg = f"Method '{method}' not implemented. Available methods: {available_methods}"
                    logger.error(error_msg)
                    raise NotImplementedError(error_msg)
                
            except Exception as session_error:
                # This catches errors from the actual MCP operations
                error_details = {
                    'error_type': type(session_error).__name__,
                    'error_message': str(session_error),
                    'method_called': method,
                    'params_used': params
                }
                logger.error(f"MCP session error: {error_details}")
                
                # Re-raise with enhanced context
                raise RuntimeError(f"MCP method '{method}' failed: {session_error}") from session_error
                
        except Exception as e:
            # Capture comprehensive error information for debugging
            error_context = {
                'method': method,
                'params': params,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'client_session_exists': self.client_session is not None,
                'traceback': traceback.format_exc()
            }
            
            logger.error(f"Comprehensive error in call_mcp_method: {error_context}")
            
            # Always re-raise so the caller can handle appropriately
            raise
    
    def schedule_async_task(self, coro):
        """
        Schedule an async task from a sync context with comprehensive error handling.
        This is like sending a request with a detailed receipt system to track what happens.
        """
        import traceback
        import concurrent.futures
        
        # Pre-flight checks - like verifying the postal system is working
        if not self.event_loop:
            error_msg = "Event loop not running - the async communication channel is down"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        
        if not self.event_loop.is_running():
            error_msg = "Event loop exists but is not running - the communication channel is stopped"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        
        if not self.client_ready.is_set():
            error_msg = "MCP client not ready - the service provider is not available"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        
        try:
            logger.info(f"Scheduling async task: {coro}")
            
            # Create the future with detailed logging
            future = asyncio.run_coroutine_threadsafe(coro, self.event_loop)
            logger.info("Async task scheduled successfully, waiting for result...")
            
            # Wait for result with a reasonable timeout
            result = future.result(timeout=20.0)  # Reduced timeout for faster feedback
            logger.info("Async task completed successfully")
            return result
            
        except concurrent.futures.TimeoutError:
            error_msg = "Async task timed out after 10 seconds - the operation took too long"
            logger.error(error_msg)
            return f"Error: {error_msg}"
            
        except concurrent.futures.CancelledError:
            error_msg = "Async task was cancelled - the operation was interrupted"
            logger.error(error_msg)
            return f"Error: {error_msg}"
            
        except Exception as e:
            # Capture the full exception context for debugging
            error_msg = f"Exception in async task: {type(e).__name__}: {str(e)}"
            full_traceback = traceback.format_exc()
            
            logger.error(f"Detailed error: {error_msg}")
            logger.error(f"Full traceback: {full_traceback}")
            
            # Return a user-friendly error message but log the technical details
            return f"Error: {error_msg}"
    
    def create_gradio_interface(self):
        """
        Create the Gradio interface that interacts with MCP client.
        This is like setting up the stage with proper sound equipment and error detection.
        """
        def handle_user_input(user_message):
            """
            Handle user input with comprehensive error reporting and debugging.
            Think of this as a customer service representative with full diagnostic tools.
            """
            import traceback
            import datetime
            
            # Create a timestamp for this interaction
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"[{timestamp}] Processing user input: '{user_message}'")
            
            try:
                # Step 1: Validate the input
                if not user_message or not user_message.strip():
                    return "Please enter a message to process."
                
                # Step 2: Check system readiness with detailed status
                logger.info("Checking system readiness...")
                
                # Check if client is ready
                if not self.client_ready.is_set():
                    logger.error("MCP client is not ready")
                    return "System Error: MCP client is still initializing. Please wait a moment and try again."
                
                # Check if event loop is running
                if not self.event_loop or not self.event_loop.is_running():
                    logger.error("Event loop is not running")
                    return "System Error: Async processing system is not available. Please restart the application."
                
                # Check if we have a client session
                if not self.client_session:
                    logger.error("No active MCP client session")
                    return "System Error: No active connection to MCP server. Please restart the application."
                
                logger.info("All system checks passed, proceeding with MCP operation...")
                
                # Step 3: Process the user input - for now, we'll list tools as an example
                # In a real implementation, you might parse the user_message to determine what action to take
                try:
                    logger.info("Attempting to list available tools...")
                    result = self.schedule_async_task(
                        self.call_mcp_method("list_tools")
                    )
                    
                    # Check if the result indicates an error (since schedule_async_task now returns error strings)
                    if isinstance(result, str) and result.startswith("Error:"):
                        logger.error(f"Async task returned error: {result}")
                        return f"Operation failed: {result}"
                    
                    # Format the successful result
                    if isinstance(result, list):
                        if len(result) == 0:
                            return "MCP server is connected, but no tools are currently available."
                        else:
                            tools_list = ", ".join(result)
                            return f"Available tools ({len(result)}): {tools_list}"
                    else:
                        return f"Received response: {result}"
                        
                except Exception as async_error:
                    # This catches errors from schedule_async_task that weren't handled there
                    error_msg = f"Async operation failed: {type(async_error).__name__}: {str(async_error)}"
                    error_trace = traceback.format_exc()
                    
                    logger.error(f"Exception in async operation: {error_msg}")
                    logger.error(f"Full async error traceback: {error_trace}")
                    
                    return f"Processing Error: {error_msg}"
                
            except Exception as e:
                # This is the outermost exception handler - it catches any errors in the UI logic itself
                error_type = type(e).__name__
                error_message = str(e)
                full_traceback = traceback.format_exc()
                
                # Log the comprehensive error information
                logger.error(f"[{timestamp}] Critical error in handle_user_input:")
                logger.error(f"  Error Type: {error_type}")
                logger.error(f"  Error Message: {error_message}")
                logger.error(f"  User Input: '{user_message}'")
                logger.error(f"  Full Traceback: {full_traceback}")
                
                # Return a user-friendly error message with enough detail for debugging
                return f"Critical Error: {error_type}: {error_message}\n\nPlease check the console logs for detailed information, or restart the application if the problem persists."
        
        # Create the Gradio interface with enhanced descriptions
        self.gradio_app = gr.Interface(
            fn=handle_user_input,
            inputs=gr.Textbox(
                label="Your message", 
                placeholder="Type your message here to interact with MCP tools...",
                lines=2
            ),
            outputs=gr.Textbox(
                label="Response", 
                lines=5,
                max_lines=10
            ),
            title="MCP Client Interface",
            description="Interact with Model Context Protocol (MCP) tools. The system will show available tools and their responses.",
            examples=[
                ["List available tools"],
                ["What can you help me with?"],
                ["Show me the tools"]
            ]
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
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
            
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status for debugging.
        This is like having a health monitor that shows all vital signs of the system.
        """
        import threading
        
        status = {
            'timestamp': str(datetime.datetime.now()),
            'event_loop': {
                'exists': self.event_loop is not None,
                'is_running': self.event_loop.is_running() if self.event_loop else False,
                'is_closed': self.event_loop.is_closed() if self.event_loop else None,
            },
            'loop_thread': {
                'exists': self.loop_thread is not None,
                'is_alive': self.loop_thread.is_alive() if self.loop_thread else False,
                'is_daemon': self.loop_thread.daemon if self.loop_thread else None,
            },
            'client': {
                'session_exists': self.client_session is not None,
                'client_ready': self.client_ready.is_set(),
                'server_params_exist': self.server_params is not None,
            },
            'gradio': {
                'app_exists': self.gradio_app is not None,
            },
            'shutdown': {
                'shutdown_event_set': self.shutdown_event.is_set(),
            },
            'threading': {
                'active_threads': threading.active_count(),
                'current_thread': threading.current_thread().name,
            }
        }
        
        return status

    def log_system_status(self):
        """Log the current system status for debugging purposes."""
        status = self.get_system_status()
        logger.info("=== SYSTEM STATUS ===")
        for category, details in status.items():
            logger.info(f"{category.upper()}: {details}:")
            # for key, value in details.items():
            # logger.info(f"  {key}: {value}")
        logger.info("=== END STATUS ===")
        
    async def test_mcp_connection(self):
        """
        Test the MCP connection and return diagnostic information.
        This is like running a connectivity test to ensure everything is working properly.
        """
        test_results = {
            'timestamp': str(datetime.datetime.now()),
            'client_session_exists': self.client_session is not None,
            'tests': {}
        }
        
        if not self.client_session:
            test_results['tests']['connection'] = {
                'status': 'FAILED',
                'error': 'No client session available'
            }
            return test_results
        
        # Test 1: Try to list tools
        try:
            tools_result = await self.client_session.list_tools()
            test_results['tests']['list_tools'] = {
                'status': 'SUCCESS',
                'tool_count': len(tools_result.tools) if tools_result.tools else 0,
                'tools': [tool.name for tool in tools_result.tools] if tools_result.tools else []
            }
        except Exception as e:
            test_results['tests']['list_tools'] = {
                'status': 'FAILED',
                'error': f"{type(e).__name__}: {str(e)}"
            }
        
        # Test 2: Try to list resources
        try:
            resources_result = await self.client_session.list_resources()
            test_results['tests']['list_resources'] = {
                'status': 'SUCCESS',
                'resource_count': len(resources_result.resources) if resources_result.resources else 0,
                'resources': [res.uri for res in resources_result.resources] if resources_result.resources else []
            }
        except Exception as e:
            test_results['tests']['list_resources'] = {
                'status': 'FAILED',
                'error': f"{type(e).__name__}: {str(e)}"
            }
        
        return test_results
            
        
    
    async def run_async(self, server_command: list, gradio_kwargs: Dict[str, Any] = None):
        """
        Main async method to run everything together.
        This is the complete performance from start to finish.
        """
        gradio_kwargs = gradio_kwargs or {}
        
        try:
            # Initialize the MCP client first
            await self.initialize_client(server_command)
            
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
    
    def run(self, server_command: list, gradio_kwargs: Dict[str, Any] = None):
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
                self.run_async(server_command, gradio_kwargs),
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
    Enhanced example usage with debugging capabilities.
    This demonstrates a production-ready setup with comprehensive monitoring.
    """
    # Create the manager instance
    manager = MCPGradioManager()
    
    # Define your MCP server command - CUSTOMIZE THIS for your specific server
    server_command = []
    
    # Example server commands for common MCP servers:
    # For a filesystem server: ["python", "-m", "mcp_filesystem", "/path/to/directory"]
    # For a database server: ["python", "-m", "mcp_database", "--db-url", "sqlite:///example.db"]
    # For a custom server: ["python", "path/to/your/server.py"]
    
    # Optional Gradio configuration
    gradio_config = {
        'server_port': 7860,
        'share': False,
        'debug': True,
        'show_error': True  # This will show detailed errors in the Gradio interface
    }
    
    # Enable more detailed logging for debugging
    logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("=== Starting MCP-Gradio Integration ===")
        
        # Log the configuration being used
        logger.info(f"Server command: {' '.join(server_command)}")
        logger.info(f"Gradio config: {gradio_config}")
        
        # Run the integrated system
        manager.run(server_command, gradio_config)
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C)")
        
        # Log system status before shutdown for debugging
        manager.log_system_status()
        
    except Exception as e:
        logger.error(f"Application error: {type(e).__name__}: {e}")
        
        # Log comprehensive error information
        import traceback
        logger.error(f"Full error traceback: {traceback.format_exc()}")
        
        # Log system status for debugging
        manager.log_system_status()
        
        # If possible, test the MCP connection to see what went wrong
        try:
            if manager.event_loop and manager.event_loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    manager.test_mcp_connection(),
                    manager.event_loop
                )
                connection_test = future.result(timeout=5.0)
                logger.info(f"MCP connection test results: {connection_test}")
        except Exception as test_error:
            logger.error(f"Could not run connection test: {test_error}")
            
    finally:
        logger.info("Application finished")
        
        # Final system status log
        try:
            manager.log_system_status()
        except Exception as status_error:
            logger.error(f"Could not log final system status: {status_error}")


# Additional helper function for debugging your specific setup
def debug_server_command(server_command):
    """
    Test if your MCP server command works before integrating with Gradio.
    This is like testing each instrument individually before the full orchestra performance.
    """
    import subprocess
    import time
    
    logger.info(f"Testing server command: {' '.join(server_command)}")
    
    try:
        # Try to start the server process
        process = subprocess.Popen(
            args = server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give the server a moment to start
        time.sleep(5)
        
        # Check if the process is still running (it should be waiting for input)
        if process.poll() is None:
            logger.info("✓ Server process started successfully and is running")
            
            # Try to send a basic MCP initialization message
            init_message = '{"jsonrpc": "2.0", "method": "initialize", "id": 1, "params": {"protocolVersion": "2024-11-05", "capabilities": {}}}\n'
            
            try:
                process.stdin.write(init_message)
                process.stdin.flush()
                
                # Wait a bit for response
                time.sleep(1)
                
                # Check if we got any response
                process.stdin.close()
                stdout, stderr = process.communicate(timeout=5)
                
                if stdout:
                    logger.info(f"✓ Server responded: {stdout[:200]}...")
                if stderr:
                    logger.warning(f"Server stderr: {stderr[:200]}...")
                    
                logger.info("Server command appears to be working correctly")
                
            except Exception as comm_error:
                logger.error(f"✗ Error communicating with server: {comm_error}")
                
        else:
            # Process exited immediately
            stdout, stderr = process.communicate()
            logger.error(f"✗ Server process exited immediately")
            logger.error(f"Exit code: {process.returncode}")
            if stdout:
                logger.error(f"Stdout: {stdout}")
            if stderr:
                logger.error(f"Stderr: {stderr}")
                
    except FileNotFoundError:
        logger.error(f"✗ Command not found: {server_command[0]}")
        logger.error("Make sure the server executable is installed and in your PATH")
        
    except Exception as e:
        logger.error(f"✗ Error testing server command: {e}")


if __name__ == "__main__":
    # Uncomment the next line to test your server command before running the full integration
    # debug_server_command(["python", ""])  # Replace with your command
    
    main()