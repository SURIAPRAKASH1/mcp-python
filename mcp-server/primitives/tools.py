import os, json, sys 
from typing import Literal
from pathlib import Path
import subprocess
from dotenv import load_dotenv
load_dotenv()  

from logging_utils import get_logger 
logger = get_logger(name = __name__)

# Import neccessary Packages. Sanity checks for libraries cause connecting mcp-client to mcp-server via stdio, client launches mcp-server as subprocess so client uses it's env libraries. If we use different env like in this case, client will invoke connection closed error and never gives any clue what's went wrong that's frustrating 😞.
try:
    from bs4 import BeautifulSoup 
    import httpx 
    import gradio as gr
except ImportError as e:
    logger.error("Got Error when Importing Packages: \n%s", e) 
    sys.exit(1) 
except Exception as e:
    logger.error("Got UnExcepted Error when Importing Packages: \n%s", e)
    sys.exit(1)

# --------------
# Configuration
#---------------
BASE_CRICKET_URL = os.environ.get("BASE_CRICKET_URL")
logger.warning("Env variable BASE_CRICKET_URL Not-Found may cause error...") if not BASE_CRICKET_URL else logger.info("")

# PR template directory 
TEMPLATES_DIR = Path(__file__).parent.parent.parent / ".github/PULL_REQUEST_TEMPLATE" or Path(__file__).parent.parent / "templates"
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


# -----------------
# Available tools
# -----------------

async def cricket_source(
    mode: Literal["live", "upcoming"], want: Literal["text", "href"] ) -> str:
    """Fetches whole html from source extracts html container that contains necessary details
    
    Args:
        mode: Which type of match do you wanna see details about.
        want: Extractor name to get details from html container.
    """

    if mode == "live":
        url = f"{BASE_CRICKET_URL}/cricket-match/live-scores"
    elif mode == 'upcoming':
        url = f"{BASE_CRICKET_URL}/cricket-match/live-scores/upcoming-matches"
    else:
        error = f"Not Implemented: Currently there's no implementation to handle {mode}. Only handels live, upcoming"
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
        if mode in ['live', 'upcoming'] and want == "text":
            text = container.get_text(separator=" ", strip= True) 
            return json.dumps({f"{mode} details": str(text)})    
        elif mode == 'live' and want == 'href':
            hrefs_list = container.find_all("a", class_ = "cb-text-link cb-mtch-lnks") 
            hrefs_string = ",".join(str(tag) for tag in hrefs_list)
            return json.dumps({"hrefs_strings": hrefs_string})
        else:
            return json.dumps({"error": f"Not Implemented for {mode} with {want}"})
        
    else:
        return json.dumps({"error": "No Available details right now!"})


async def fetch_cricket_details(mode: Literal["live", "upcoming"])-> str:
    """Get cricket Live or Upcomming match details

    Args:
        mode : Either "live" or "upcoming"
    """
    response = await cricket_source(mode.strip().lower(), want= 'text')
    return response


async def live_cricket_scorecard_href()-> str:
    """String of comma separated anchor tags with href attributes that pointing to live cricket scorecards """
    response = await cricket_source('live', 'href')
    return response


async def live_cricket_scorecard(href: str)-> str:
    """Get Live cricket match scorecard details.
    (e.g, href = "/live-cricket-scorecard/119495/cd-vs-hbh-7th-match-global-super-league-2025")

    Args:
        href: live cricket match scorecard endpoint
    """
    scorecard_url = f"{BASE_CRICKET_URL}{href}"

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


async def analyze_file_changes(
    base_branch: str = "main",
    include_diff: bool = True,
    max_diff_lines: int = 400,
    working_directory: str = ""        # Optional[str] gradio will give error
    ) -> str:
    """Get the full diff and list of changed files in the current git repository.
    
    Args:
        base_branch: Base branch to compare against (default: main)
        include_diff: Include the full diff content (default: true)
        max_diff_lines: Maximum number of diff lines to include (default: 400)
        working_directory: Directory to run git commands in (default: current directory)
    """
    try:
        
        # Use provided working directory or current directory
        cwd = working_directory if working_directory else os.getcwd()
        
        # Debug output
        debug_info = {
            "provided_working_directory": working_directory,
            "actual_cwd": cwd,
            "server_process_cwd": os.getcwd(),
            "server_file_location": str(Path(__file__).parent),
        }
        
        # Get list of changed files
        files_result = subprocess.run(
            ["git", "diff", "--name-status", f"{base_branch}...HEAD"],
            stdin= subprocess.DEVNULL, 
            stdout= subprocess.PIPE, 
            stderr = subprocess.PIPE,
            # capture_output=True,
            text=True,
            check=True,
            cwd=cwd
        )
        
        # Get diff statistics
        stat_result = subprocess.run(
            ["git", "diff", "--stat", f"{base_branch}...HEAD"],
            stdin= subprocess.DEVNULL, 
            stdout= subprocess.PIPE, 
            stderr = subprocess.PIPE,
            # capture_output=True,
            text=True,
            cwd=cwd
        )
        
        # Get the actual diff if requested
        diff_content = ""
        truncated = False
        if include_diff:
            diff_result = subprocess.run(
                ["git", "diff", f"{base_branch}...HEAD"],
                stdin= subprocess.DEVNULL, 
                stdout= subprocess.PIPE, 
                stderr = subprocess.PIPE,
                # capture_output=True,
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
            stdin= subprocess.DEVNULL, 
            stdout= subprocess.PIPE, 
            stderr = subprocess.PIPE,
                # capture_output=True,
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

# Metadata for Gradio components
TOOL_COMPONENTS = {
    "cricket_source": {            
        "is_gradio_api": True,
    },
    "fetch_cricket_details": {
        "is_gradio_api": False, 
        "inputs": gr.Radio(["live", "upcoming"], label="Mode"),
        "outputs": gr.JSON(label= "Details"), 
    },
    "live_cricket_scorecard_href": {
        "is_gradio_api": False, 
        "inputs": None, 
        "outputs": gr.JSON(label = 'Scorecard hrefs')
    },
    "live_cricket_scorecard": {
        "is_gradio_api": True,    
    },
    "analyze_file_changes": {
        "is_gradio_api": True, 
    }, 
    "get_pr_templates": {
        "is_gradio_api": False, 
        "inputs": None,  # if None gradio will automatically create Generate button for us
        "outputs": gr.JSON(label= "Available PR templates")
    }, 
    "suggest_template": {
        "is_gradio_api": True
    }
}
