import logging, os, json, sys 
from typing import Any, Literal, Optional 
from pathlib import Path
import subprocess
from dotenv import load_dotenv
load_dotenv()  

# -----------
# Logging
# ------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

fmt = "%(asctime)s -- %(levelname)s -- %(name)s -- %(message)s"
file_handler = logging.FileHandler("multitools-server.log")
file_handler.setFormatter(logging.Formatter(fmt))

logger.addHandler(file_handler)

# Now import neccessary Packages
try:
    from mcp.server.fastmcp import FastMCP
    from bs4 import BeautifulSoup 
    import httpx 
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

# --------------
# Configuration
#---------------
BASE_CRICKET_URL = os.environ.get("BASE_CRICKET_URL")
logger.warning("Env variable BASE_CRICKET_URL Not-Found may cause error...") if not BASE_CRICKET_URL else logger.info("")

# PR template directory 
TEMPLATES_DIR = Path(__file__).parent / "templates"
logger.warning("TEMPLATES_DIR Not-Found may cause Error...") if not TEMPLATES_DIR else logger.info("TEMPLATES_DIR: \n%s", TEMPLATES_DIR)

# Default PR templates
DEFAULT_TEMPLATES = {
    "bug.md": "Bug Fix",
    "feature.md": "Feature",
    "docs.md": "Documentation",
    "refactor.md": "Refactor",
    "test.md": "Test",
    "performance.md": "Performance",
    "security.md": "Security"
}

# Type mapping for PR templates
TYPE_MAPPING = {
    "bug": "bug.md",
    "fix": "bug.md",
    "feature": "feature.md",
    "enhancement": "feature.md",
    "docs": "docs.md",
    "documentation": "docs.md",
    "refactor": "refactor.md",
    "cleanup": "refactor.md",
    "test": "test.md",
    "testing": "test.md",
    "performance": "performance.md",
    "optimization": "performance.md",
    "security": "security.md"
}


# ----------------------
# Available tools for LLM
# -----------------------

async def cricket_source(mode: str, want: str) -> str:
    """Fetches whole html from source url then extracts html container that contains necessary details"""

    if mode == "live":
        url = f"{BASE_CRICKET_URL}/cricket-match/live-scores"
    elif mode == 'upcomming':
        url = f"{BASE_CRICKET_URL}/cricket-match/live-scores/upcoming-matches"
    else:
        error = f"Not Implemented: Currently there's no implementation to handle {mode}. Only handels live, upcomming"
        return json.dumps({"error": error})
        
    try:
        async with httpx.AsyncClient(timeout= 10.0) as client:
            response = await client.get(url= url) 
            response.raise_for_status()                    # if ain't 2xx it will raise HTTP error
    except httpx.HTTPError as e:
        return json.dumps({'error': str(e)})
    except Exception as e:
        return json.dumps({'error': str(e)})

    if response:
        # convert htmldoc content to proper html form using bs
        html = BeautifulSoup(response.content, "html.parser")

        # find where the content is
        container = html.find("div", class_= 'cb-col cb-col-100 cb-rank-tabs')
        if mode in ['live', 'upcomming'] and want == "text":
            text = container.get_text(separator=" ", strip= True) 
            return json.dumps({"output": str(text)})    
        elif mode == 'live' and want == 'herf':
            herfs_list = container.find_all("a", class_ = "cb-text-link cb-mtch-lnks") 
            herfs_string = ",".join(str(tag) for tag in herfs_list)
            return json.dumps({"output": herfs_string})
        else:
            return json.dumps({"error": f"Not Implemented for {mode} with {want}"})
        
    else:
        return json.dumps({"error": "No Available details right now!"})

@mcp.tool()
async def fetch_cricket_details(mode: Literal["live", "upcomming"])-> str:
    """ Get cricket Live or Upcomming match details
    Args:
        mode : Either "live" or "upcomming"
    """
    response = await cricket_source(mode.strip().lower(), want= 'text')
    return response


@mcp.tool()
async def live_cricket_scorecard_herf()-> str:
    """String of comma separated anchor tags contains herf attributes that pointing to live cricket scorecards """
    response = await cricket_source('live', 'herf')
    return response


@mcp.tool()
async def live_cricket_scorecard(herf: str)-> str:
    """Live cricket match scorecard details for given herf.
    (e.g, herf = "/live-cricket-scorecard/119495/cd-vs-hbh-7th-match-global-super-league-2025")

    Args:
        herf: live cricket match scorecard endpoint
    """
    scorecard_url = f"{BASE_CRICKET_URL}{herf}"

    try:
        async with httpx.AsyncClient(timeout= 10.0) as client:
            response = await client.get(url = scorecard_url)
            response.raise_for_status()
    except httpx.HTTPError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({'error': str(e)})
    
    # extract html container
    if response:
        html = BeautifulSoup(response.content, "html.parser")
        live_scorecard = html.find("div", timeout = "30000")
        details = live_scorecard.get_text(separator=" ", strip=True)
        return json.dumps({'output': str(details)})
    else:
        return json.dumps({'error': "No Available details right now"})


@mcp.tool()
async def analyze_file_changes(
    base_branch: str = "main",
    include_diff: bool = True,
    max_diff_lines: int = 400,
    working_directory: Optional[str] = None
    ) -> str:
    """Get the full diff and list of changed files in the current git repository.
    
    Args:
        base_branch: Base branch to compare against (default: main)
        include_diff: Include the full diff content (default: true)
        max_diff_lines: Maximum number of diff lines to include (default: 400)
        working_directory: Directory to run git commands in (default: current directory)
    """
    try:
        # Try to get working directory from roots first
        if working_directory is None:
            try:
                context = mcp.get_context()
                roots_result = await context.session.list_roots()
                # Get the first root - Claude Code sets this to the CWD
                root = roots_result.roots[0]
                # FileUrl object has a .path property that gives us the path directly
                working_directory = root.uri.path
            except Exception:
                # If we can't get roots, fall back to current directory
                pass
        
        # Use provided working directory or current directory
        cwd = working_directory if working_directory else os.getcwd()
        
        # Debug output
        debug_info = {
            "provided_working_directory": working_directory,
            "actual_cwd": cwd,
            "server_process_cwd": os.getcwd(),
            "server_file_location": str(Path(__file__).parent),
            "roots_check": None
        }
        
        # Add roots debug info
        try:
            context = mcp.get_context()
            roots_result = await context.session.list_roots()
            debug_info["roots_check"] = {
                "found": True,
                "count": len(roots_result.roots),
                "roots": [str(root.uri) for root in roots_result.roots]
            }
        except Exception as e:
            debug_info["roots_check"] = {
                "found": False,
                "error": str(e)
            }
        
        # Get list of changed files
        files_result = subprocess.run(
            ["git", "diff", "--name-status", f"{base_branch}...HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd
        )
        
        # Get diff statistics
        stat_result = subprocess.run(
            ["git", "diff", "--stat", f"{base_branch}...HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd
        )
        
        # Get the actual diff if requested
        diff_content = ""
        truncated = False
        if include_diff:
            diff_result = subprocess.run(
                ["git", "diff", f"{base_branch}...HEAD"],
                capture_output=True,
                text=True,
                cwd=cwd
            )
            diff_lines = diff_result.stdout.split('\n')
            
            # Check if we need to truncate
            if len(diff_lines) > max_diff_lines:
                diff_content = '\n'.join(diff_lines[:max_diff_lines])
                diff_content += f"\n\n... Output truncated. Showing {max_diff_lines} of {len(diff_lines)} lines ..."
                diff_content += "\n... Use max_diff_lines parameter to see more ..."
                truncated = True
            else:
                diff_content = diff_result.stdout
        
        # Get commit messages for context
        commits_result = subprocess.run(
            ["git", "log", "--oneline", f"{base_branch}..HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd
        )
        
        analysis = {
            "base_branch": base_branch,
            "files_changed": files_result.stdout,
            "statistics": stat_result.stdout,
            "commits": commits_result.stdout,
            "diff": diff_content if include_diff else "Diff not included (set include_diff=true to see full diff)",
            "truncated": truncated,
            "total_diff_lines": len(diff_lines) if include_diff else 0,
            "_debug": debug_info
        }
        
        return json.dumps(analysis, indent=2)
        
    except subprocess.CalledProcessError as e:
        return json.dumps({"error": f"Git error: {e.stderr}"})
    except Exception as e:
        return json.dumps({"error": str(e)})
  


@mcp.tool()
async def get_pr_templates() -> str:
    """List available PR templates with their content."""
    templates = [
        {
            "filename": filename,
            "type": template_type,
            "content": (TEMPLATES_DIR / filename).read_text()
        }
        for filename, template_type in DEFAULT_TEMPLATES.items()
    ]
    
    return json.dumps(templates, indent=2)



@mcp.tool()
async def suggest_template(changes_summary: str, change_type: str) -> str:
    """Let LLM analyze the changes and suggest the most appropriate PR template.
    
    Args:
        changes_summary: Your analysis of what the changes do
        change_type: The type of change you've identified (bug, feature, docs, refactor, test, etc.)
    """
    
    # Get available templates
    templates_response = await get_pr_templates()
    templates = json.loads(templates_response)
    
    # Find matching template
    template_file = TYPE_MAPPING.get(change_type.lower(), "feature.md")
    selected_template = next(
        (t for t in templates if t["filename"] == template_file),
        templates[0]  # Default to first template if no match
    )
    
    suggestion = {
        "recommended_template": selected_template,
        "reasoning": f"Based on your analysis: '{changes_summary}', this appears to be a {change_type} change.",
        "template_content": selected_template["content"],
        "usage_hint": "LLM can help you fill out this template based on the specific changes in your PR."
    }
    
    return json.dumps(suggestion, indent=2)

if __name__ == "__main__":

    transport = "stdio"
    if transport != "streamable-http":
        logger.info("multitools-server is started 🚀🚀🚀")
    else:
        print("multitools-server is started 🚀🚀🚀")
    mcp.run(transport = transport)