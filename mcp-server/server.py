import os, sys 
import argparse
from dotenv import load_dotenv 
load_dotenv()

from logging_utils import get_logger 
logger = get_logger(name = __name__)


try:
    from mcp.server.fastmcp import FastMCP
    from registry import get_tool_functions
except ImportError as e:
    logger.error("Got Error when Importing Packages: \n%s", e) 
    sys.exit(1) 
except Exception as e:
    logger.error("Got UnExcepted Error when Importing Packages: \n%s", e)
    sys.exit(1)

# -------------------------
# Initiating FastMCP server 
# -------------------------
mcp = FastMCP("multitools-server") 


# Auto register all tools 
for name, func in get_tool_functions().items(): 
    mcp.tool(name)(func) 


if __name__ == "__main__":
    # Argument parser to handle CLI args
    parser = argparse.ArgumentParser() 

    parser.add_argument("--transport", type= str, default = "stdio", help= "Which transport type do you want to run mcp server?. Controlled by --transport cli arg OR TRANSPORT env. If didn't provide either of those default to stdio.") 
    args = parser.parse_args() 

    transport = os.environ.get("TRANSPORT") or args.transport
    
    logger.info(f"multitools-server is started via {transport} transport")
    mcp.run(transport = transport)