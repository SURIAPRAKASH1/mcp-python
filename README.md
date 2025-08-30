# _MCP-PYTHON:_

_Model Context Protocal Implemenation in Pythonic way.It's small project that embracess mcp.Am working on it ....._

# _Content:_

_Note: Am just walking through project's Structure and Explaning what's it? How's is it used? in the universe of MCP._

## [mcp-client](mcp-client):

host: Host an application/UI; Anyone (enduser) can interact with. Gradio act as host.
client: Client wraped by host; Initiate communication with server behalf of Host.

## mcp-server:

primitives: Core components for MCP. 1. tools
server: An piece of code listening for client
app: MCP server with gradio integration.
