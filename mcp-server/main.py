from typing import Any
from mcp.server.fastmcp import FastMCP 
import httpx 

import logging, os
from dotenv import load_dotenv
load_dotenv()


# -----------
# Logging
# ------------
logger = logging.getLogger(__name__) 

# formatter
fmt = logging.Formatter("%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s")

# handlers 
console_handler = logging.StreamHandler() 
file_handler = logging.FileHandler() 

# add to logger 
logger.addHandler(console_handler) 
logger.addHandler(file_handler.setFormatter(fmt))


# --------------
# Configuration
#---------------
BASE_CRICKET_URI = os.environ.get("BASE_CRICKET_URI", "False")


# -------------------------
# Initiating FastMCP server 
# -------------------------
mcp = FastMCP("multitools-server") 


# ----------------------
# Available tools for LLM
# -----------------------