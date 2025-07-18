import asyncio
import httpx
from contextlib import AsyncExitStack
from mcp.client.session import ClientSession

class MCPClient():
    """ MCPClient: Client that will connect to MCP server and performers client-server communication"""

    def __init__(self):
        
        self.session[ClientSession] = None 
        self.exit_stack = AsyncExitStack() 
    
    async def connect_to_server(self):
        pass


async def main():
    client = MCPClient()


if __name__ == "__main__":
    asyncio.run(main())
