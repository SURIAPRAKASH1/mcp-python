try:
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.session import ClientSession
    import json

    from .format import Formatter
except ImportError:
    raise ImportError

async def select_relevant_tools(url: str, query: str, tools_list: list[dict], top_k: int = 3, cache_tool_embs: bool = True):
    async with streamablehttp_client(url=url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            fn_name = "query_tools_similarity_find_similar_tools"
            fn_args = {
                "query": query,
                "tools_list": json.dumps(tools_list),
                "top_k": top_k,
                "cache_tool_embs": cache_tool_embs
            }
            result = await session.call_tool(name=fn_name, arguments=fn_args)
            # Parse and return the result as needed
            return Formatter.tool_parser(result.content[1].text)