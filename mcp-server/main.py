from mcp.server.fastmcp import FastMCP 
import httpx 

from typing import Any, Literal
from bs4 import BeautifulSoup
import logging, os, json
from dotenv import load_dotenv
load_dotenv()


# -----------
# Logging
# ------------
logger = logging.getLogger(__name__) 
logger.setLevel(logger.debug)

# formatter
fmt = logging.Formatter("%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s")

# handlers 
# console_handler = logging.StreamHandler() 
file_handler = logging.FileHandler(filename= "multitools-server.log")

# add to logger 
# logger.addHandler(console_handler) 
logger.addHandler(file_handler.setFormatter(fmt))


# --------------
# Configuration
#---------------
BASE_CRICKET_URL = os.environ.get("BASE_CRICKET_URI", "False")


# -------------------------
# Initiating FastMCP server 
# -------------------------
mcp = FastMCP("multitools-server") 


# ----------------------
# Available tools for LLM
# -----------------------

async def cricket_source(mode: str) -> str:
    """Fetches whole html from source url then extracts html container that contains necessary details"""

    if mode == "live":
        url = f"{BASE_CRICKET_URL}/cricket-match/live-scores"
    elif mode == 'upcomming':
        url = f"{BASE_CRICKET_URL}/cricket-match/live-scores/upcoming-matches"
    else:
        error = f"Not Implemented: Currently there's no implementation to handle {mode}. Only handels live, upcomming"
        logger.error(msg= error)
        return json.dumps({"error": error})
        
    try:
        async with httpx.AsyncClient(timeout= 10.0) as client:
            response = await client.get(url= url) 
            response.raise_for_status()                    # if not 2xx it will raise HTTP error
    except httpx.HTTPError as e:
        logger.error("\n%s", e) 
        return json.dumps({'error': str(e)})
    except Exception as e:
        logger.error("\n%s", e) 
        return json.dumps({'error': str(e)})

    if response:
        # convert htmldoc content to proper html form using bs
        html = BeautifulSoup(response.content, "html.parser")
        # find where the content is
        content = html.find("div", class_= 'cb-col cb-col-100 cb-rank-tabs')
        return content
    else:
        return json.dumps({"error": "No Available details right now!"})

@mcp.tool()
async def fetch_live_cricket_details(mode: Literal["live", "upcomming"])-> str:
    """ Get cricket live or upcomming match details
    Args:
        mode : Either "live" or "upcomming"
    """

    response = await cricket_source(mode.strip().lower())
    if response['error']:
        return response
    live_details = response.get_text(separator = "\n", strip = True)
    return json.dumps({'output': str(live_details)}) 

@mcp.resource("resource://scorecard_herf")
async def live_cricket_scorecard_herf()-> str:
    """Returns string of comma separated anchor tags contains herf attributes that pointing to live cricket scorecards """

    response = await cricket_source("live")
    if response['error']:
        return response
    herfs_list = response.find_all("a", class_ = "cb-text-link cb-mtch-lnks")
    herfs_string = ",".join(str(tag) for tag in herfs_list)
    return json.dumps({'output': herfs_string}) 


@mcp.tool()
async def live_cricket_scorecard(herf: str)-> str:
    """Live cricket match scorecard details for given herf.
    (e.g, herf = "/live-cricket-scorecard/119495/cd-vs-hbh-7th-match-global-super-league-2025")

    Args:
        herf (str): herf for scorescard endpoint
    """
    scorecard_url = f"{BASE_CRICKET_URL}{herf}"

    try:
        with httpx.AsyncClient(timeout= 10.0) as client:
            response = client.get(url = scorecard_url)
            response.raise_for_status()
    except httpx.HTTPError as e:
        logger.error("\n%s", e) 
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("\n%s", e) 
        return json.dumps({'error': str(e)})
    
    # extract html container
    if response:
        html = BeautifulSoup(response.content, "html.parser")
        live_scorecard = html.find("div", timeout = "30000")
        details = live_scorecard.get_text(separator="\n", strip=True)
        return json.dumps({'output': str(details)})
    else:
        return json.dumps({'error': "No Available details right now"})

    