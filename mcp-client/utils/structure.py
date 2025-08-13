from typing import List, Dict, Any


class Structure:
    "Utily class for Structuring LoT"

    @classmethod
    def prepare_chat_messages(cls, message: str, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepares User and Assistant chat history in a structure way"""

        # History of user - assistant chat
        messages = []
        if history:
            # go through all conversation extract user & assistant 
            for hist in history:
                if hist.get("role") == "user":
                    messages.append({"role": "user", "content": hist.get("content")})
                elif hist.get("role") == "assistant":
                    messages.append({"role": "assistant", "content": hist.get("content")})

        # finaly append current user query to messages
        messages.append({"role": "user", "content": message})
        return messages

    @classmethod
    def prepare_default_tool_definition(cls, tool_definition: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepares Default Tool Definition in Structure Way.

        Args:
            tool_definition: MCP Raw tool definition
        """
        tools =  [{
                    "name": tool.name, 
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                    } for tool in tool_definition]

        return tools
