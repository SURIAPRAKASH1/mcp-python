import os
from typing import Literal, Optional, List, Any, Callable, Dict
from dotenv import load_dotenv
load_dotenv()

try:
    from huggingface_hub import InferenceClient 
    from google.genai import Client
    from google.genai import types
    import httpx 
    from httpx import AsyncClient

    from .tool_definition import google_definition, huggingface_definition
except ImportError:
    raise ImportError


HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN", "False") 
GOOGLE_GEMINI_API_TOKEN = os.environ.get("GOOGLE_GEMINI_API_TOKEN", "False")
NGROK_URL = os.environ.get("NGROK_URL", "False")

class Chat:
    "Single instance to Process User query via differnt Providers/Models"

    def __init__(self, plateform: Literal["huggingface", "google", "notebook"]) -> None:
        """Preapring Model from providers for processing User query.

        Args: 
            plateform: Where should get model from ?
        """

        self.plateform: str = plateform
        self.client: Optional[object] =  None

        # model from huggingface
        if self.plateform == "huggingface" and HUGGINGFACE_API_TOKEN != "False":
            self.client = InferenceClient(
                api_key= HUGGINGFACE_API_TOKEN, 
                provider= "auto" 
            )
        # model from google 
        elif self.plateform == "google" and GOOGLE_GEMINI_API_TOKEN != "False":
            self.client = Client(
             api_key= GOOGLE_GEMINI_API_TOKEN   
            )
        elif self.plateform == "notebook":
            self.client = "Custom"
        else:
            return NotImplementedError(self.plateform)
        
    def google(self, 
               call_tool: Callable[[Dict, Optional[float]], Dict],
               tool_model: str,
               response_model: Optional[str],
               messages: List[dict[str, Any]], 
               tools: List[dict[str, Any]]) -> str:
        """Processing messages and tools via Google Models.

        Args:
            call_mcp_method: Function utilization for mcp methods.
            tool_model: Google's (gemini/gemma) model Name that has tool calling functionality.
            response_model: If response_model (may not have tool calling functionality) Name is given then tool_model and response used to ans query.
            messages: All conversations in openai style format.
            tools: All Available tools.
        """ 
        
        # Config tools definition to google compatible format 
        mcp_tools = types.Tool(function_declarations=[google_definition(tool) for tool in tools])
        config = types.GenerateContentConfig(tools = [mcp_tools]) 

        # Define user prompt in google compatible format
        contents = [] 
        for message in messages:
            if message.get("role") == "user":
                contents.append(
                    types.Content(
                        role = "user", 
                        parts = [
                            types.Part(text = message.get("content"))
                        ]
                    )
                )
            elif message.get("role") == "assistant":
                contents.append(
                    types.Content(
                        role = "model", 
                        parts = [
                            types.Part(text = message.get("content"))
                        ]
                    )
                )
            else:
                raise NotImplementedError(message) 

       # Call model with function declaration 
        response = self.client.models.generate_content(
            model = tool_model,
            contents= contents, 
            config= config
        )

        final_text = [] 
        function_response_parts = []

        # It's ain't good approach to handle text will be sometime's CoT
        # On that cause model will use tools as well that's why 2 if. 
        if response.text:                                      # if model reponsed with it's knowledge
            final_text.append(response.text) 
            # contents.append(response.candidates[0].content)    # it usually don't need but for sanity
        if response.function_calls:                          # True even for single function call 
            contents.append(response.candidates[0].content)    # send back what model does

            # Process each function calls as model requested
            for fn in response.function_calls:
                tool_name = fn.name
                tool_args = fn.args 

                # Execute tool call
                print(f"Calling function {tool_name} with args {tool_args}")
                tool_call_response = call_tool(params = {"tool_name": tool_name, "arguments": tool_args}, timeout = 30)
                if tool_call_response['error']:
                    return  f"❌ Error When calling tool {tool_name} with arguments {tool_args}: \n{tool_call_response}"
                # Wrap it as a Part so the LLM can later consume it
                part = types.Part.from_function_response(
                    name= tool_name,
                    response= tool_call_response["result"]
                )
                function_response_parts.append(part)

            # Create ONE User-role message that bundles *all* those parts:
            contents.append(
                types.Content(
                    role="user", 
                    parts=function_response_parts
                )
            )

            # Again invoke model with results of function
            final_response = self.client.models.generate_content(
                model = response_model or tool_model , 
                # config = config, 
                contents = contents
            )

            final_text.append(final_response.text)

        return "\n".join(final_text)    

    async def custom(self, 
                    call_tool: Callable[[Dict, Optional[float]], Dict],
                    messages: List[dict[str, Any]], 
                    tools: List[dict[str, Any]]):
        
        # some changes in tools define to compatible with render
        mcp_tools = [huggingface_definition(tool) for tool in tools]

        # make copy of privious conversations(just a history)
        history = messages.copy()
        tool_calls = []      # if model decided to invoke tool call we keep track of tool info

        headers = {"Accept": "text/event-stream"}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=None)) as client:
                async with client.stream("POST", NGROK_URL, json={"messages": messages, "tools": mcp_tools}, headers=headers) as resp:
                    resp.raise_for_status()
                    print("Connected. status:", resp.status_code, "headers:", dict(resp.headers))

                    buffer = ""
                    current_channel = None
                    current_content = ""
                    streaming = False
                    tool_call = False

                    # ONE TOKEN AT A TIME
                    async for line in resp.aiter_lines():
                        if line is None:
                            break
                        if not line:
                            continue
                        if line.startswith("data:"):
                            payload = line[len("data:"):].strip()
                            if payload == "[DONE]":
                                print("\n[STREAM DONE]")
                                break

                            # handle payload to get token. Here we're hard coding cause LLM api is created by us 😀
                            payload = line[len("data: "):].strip("\n\n")
                            buffer += payload    

                            # Detect channel start when not currently streaming
                            if not streaming:
                                start_idx = buffer.find("<|channel|>")
                                if start_idx != -1:
                                    msg_idx = buffer.find("<|message|>", start_idx)
                                    to_idx = buffer.find(" to=", start_idx)
                                    constrain_idx = buffer.find("<|constrain|>", start_idx)
                                    call_idx = buffer.find("<|call|>", start_idx)

                                    # plain channel + message (no tool)
                                    if msg_idx != -1 and (to_idx == -1 and constrain_idx == -1 and call_idx == -1):
                                        current_channel = buffer[start_idx + len("<|channel|>") : msg_idx].strip()
                                        buffer = buffer[msg_idx + len("<|message|>") :]
                                        streaming = current_channel in ("final", "analysis")
                                        current_content = ""
                                    else:
                                        # tool-call branch — be defensive about indexes
                                        if to_idx != -1 and constrain_idx != -1 and call_idx != -1 and msg_idx != -1:
                                            current_channel = buffer[start_idx + len("<|channel|>") : to_idx].strip()
                                            tool_call = current_channel == "commentary"
                                            func = buffer[to_idx + len(" to=") : constrain_idx].strip()
                                            # safe split
                                            parts = func.split(".")
                                            func_name = parts[1] if len(parts) > 1 else None
                                            func_args = buffer[msg_idx + len("<|message|>") : call_idx].strip()
                                            if tool_call and func_name and func_args:
                                                tool_calls.append({"name": func_name, "args": func_args})
                                                # advance buffer past the <|call|> marker
                                                buffer = buffer[call_idx + len("<|call|>") :]
                                            else:
                                                # malformed tool-call payload
                                                # yield an error message (string) and stop streaming gracefully
                                                yield ("❌ Malformed tool call payload", None)
                                                return
                                        else:
                                            # not enough markers yet — keep buffering
                                            pass

                            # Stream content
                            if streaming:
                                end_idx = buffer.find("<|end|>")
                                return_idx = buffer.find("<|return|>")
                                if end_idx == -1 and return_idx == -1:
                                    # No end or return yet, stream all available
                                    chunk = buffer
                                    current_content += chunk
                                    yield chunk, current_channel
                                    buffer = ""
                                elif end_idx != -1 and return_idx == -1:
                                    # End found, stream up to end
                                    chunk = buffer[:end_idx]
                                    current_content += chunk
                                    yield chunk, current_channel
                                    buffer = buffer[end_idx+len("<|end|>"): ]
                                    # Save to history
                                    if current_channel == "final":
                                        history.append({"role": "assistant", "content": current_content})
                                    elif current_channel == "analysis":
                                        history.append({"role": "assistant", "content": current_content, "thinking": ""})
                                    else:
                                        yield (f"❌ Not valid channel: {current_channel}", None)
                                        return
                                    # Reset for next message
                                    current_channel = None
                                    current_content = ""
                                    streaming = False
                                elif return_idx != -1 and end_idx == -1:
                                    # Return found, stream up return
                                    chunk = buffer[:return_idx]
                                    current_content += chunk
                                    yield chunk, current_channel
                                    buffer = buffer[return_idx+len("<|return|>"): ]
                                    # Save to history
                                    if current_channel == "final":
                                        history.append({"role": "assistant", "content": current_content})
                                    elif current_channel == "analysis":
                                        history.append({"role": "assistant", "content": current_content, "thinking": ""})
                                    else:
                                        yield (f"❌ Not valid channel: {current_channel}", None)
                                        return
                                    # Reset for next message
                                    current_channel = None
                                    current_content = ""
                                    streaming = False
                                
                        else:
                            buffer += line

        except httpx.HTTPStatusError as exc:
            yield f"❌ HTTP Error: {exc.response.status_code}, {exc}"
            return
        except httpx.RequestError as exc:
            yield f"❌ Request error: {exc}"
            return
        except StopAsyncIteration:
            # defensive: if something internally raised StopAsyncIteration, end gracefully
            return
        except Exception as exc:
            yield f"❌ Unexpected error: {exc}"
            return

        # Have to process tool calls and then invoke model again for final ans
        if tool_calls:
            for tool in tool_calls:
                tool_name = tool["name"]
                tool_args = tool["args"]
                yield f"Calling tool {tool_name} with arguments {tool_args}", current_channel
                #----Here Tool calling------#

