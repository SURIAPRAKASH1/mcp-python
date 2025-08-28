from typing import List, Dict, Any
import json, re

class Formatter:
    "Utily class for Formatting LoT"

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
    
    @classmethod
    def format_table_content(table_buffer: str) -> str:
        """
        Format table content for better display.

        Args:
            table_buffer: Complete buffer contains table chunk
        """
        # Handle JSON tables
        if '<TABLE_JSON>' in table_buffer:
            json_content = table_buffer.replace('<TABLE_JSON>', '').replace('</TABLE_JSON>', '').strip()
            try:
                # Parse and reformat JSON for better readability
                parsed_json = json.loads(json_content)
                formatted_json = json.dumps(parsed_json, indent=2)
                return f"```json\n{formatted_json}\n```\n\n"
            except:
                # If parsing fails, return as code block
                return f"```json\n{json_content}\n```\n\n"
        
        # Handle markdown tables (already formatted, just return as-is)
        elif '|' in table_buffer:
            return table_buffer + "\n"
        
        # Handle other table formats
        else:
            return table_buffer + "\n"

    def is_table_start(chunk: str) -> bool:
        """Detect various table format beginnings."""
        # Look for markdown table, HTML table, fenced code block, or JSON markers
        return bool(re.match(r'^\s*(\|.+\|)', chunk)) or \
            '<table' in chunk.lower() or \
            chunk.strip().startswith('```') or \
            '<TABLE_JSON>' in chunk or \
            (chunk.strip().startswith('[') and '{' in chunk) or \
            (chunk.strip().startswith('{') and '"' in chunk)

    def is_table_end(buffer: str, last_chunk: str) -> bool:
        """Determine when a table is complete using various heuristics."""
        
        # HTML table closing
        if '</table>' in buffer.lower() or '</TABLE_JSON>' in buffer:
            return True
        
        # Markdown table: blank line after table rows
        if last_chunk.strip() == '' and '|' in buffer and buffer.count('\n') > 1:
            return True
            
        # Code block closing
        if last_chunk.strip().endswith('```'):
            return True
        
        # JSON completion: try parsing if buffer looks like JSON
        json_content = buffer.strip()
        if json_content.startswith('<TABLE_JSON>'):
            json_content = json_content.replace('<TABLE_JSON>', '').replace('</TABLE_JSON>', '').strip()
        
        if json_content.startswith(('{', '[')):
            try:
                json.loads(json_content)
                return True
            except:
                pass
        
        return False