import os
from typing import Literal, Optional, List, Any
from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import InferenceClient 
from google.genai import Client
from google.genai import types
from mcp.client.session import ClientSession

from .tool_definition import google_definition, huggingface_definition


HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN", "False") 
GOOGLE_GEMINI_API_TOKEN = os.environ.get("GOOGLE_GEMINI_API_TOKEN", "False")

class Chat:
    "Single instance to Process User query via differnt Providers/Models"

    def __init__(self, plateform: Literal["huggingface", "google"]) -> None:
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
        else:
            raise NotImplementedError(self.plateform)
        
    async def google(self, 
               session: ClientSession,
               tool_model: str,
               response_model: Optional[str],
               messages: List[dict[str, Any]], 
               tools: List[dict[str, Any]]) -> str:
        """Processing messages and tools via Google

        Args:
            session: MCP ClientSession.
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

        if response.text:                                      # if model reponsed with it's knowledge
            final_text.append(response.text) 
            contents.append(response.candidates[0].content)    # it usually don't need but for sanity
        elif response.function_calls:                          # True even for single function call 
            contents.append(response.candidates[0].content)    # send back what model does

            # Process each function calls as model requested
            for fn in response.function_calls:
                tool_name = fn.name
                tool_args = fn.args 

                # Exceute tool call
                print(f"Calling function {tool_name} with args {tool_args}")
                result = await session.call_tool(name = tool_name, arguments= tool_args) 
                print(f"tool call response format : \n{result}")
                # Wrap it as a Part so the LLM can later consume it
                part = types.Part.from_function_response(
                    name= tool_name,
                    response={"result": result.content}
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
                config = config, 
                contents = contents
            )

            final_text.append(final_response.text)

        return "\n".join(final_text)    

    async def custom(self):
        
        # headers = {"user-agent": "mcp-client/0.0.1","content-type": "application/json"}

        # async with httpx.AsyncClient(timeout= 10.0) as client:
        #     try:
        #         to_llm_context = {"messages": messages, "tools": available_tools}
        #         llm_response = await client.post(url = GEMMA3N_URI, headers= headers, json= to_llm_context)
        #         llm_response.raise_for_status()            
        #     except httpx.HTTPStatusError as exc:
        #         return f"Error Response {exc.response.status_code} while requesting {exc.request.url}"
        #     except httpx.RequestError as exc:
        #         return f"Error while requesting {exc.request.url}"
        #     except Exception as exc:
        #         return f"Unexpected Error when Accesssing LLM API: \n {exc}"

        # final_text = []
        # assistant_message_context = []
        # response = llm_response.get('message')

        # if response['content']:                       
        #     final_text.append(response['content'])
        #     assistant_message_context.append(response['content'])
        # elif response['tool_calls']:

        #     for tool_call in response['tool_calls']:
        #         function_name, function_args = tool_call['function'].get('name'), tool_call['function'].get("arguments")

        #         # Execute tool call
        #         result = self.session.call_tool(function_name, function_args)
        #         print(f"Calling tool {function_name} with arguments {function_args}")
                
        #         messages.append({
        #             'role': "assistant",
        #             'content': tool_call['function']
                
        #         }) 
        #         # add tool result to LLM's context
        #         messages.append({
        #             "role": "tool",
        #             "content": result.content
        #         })

        #         # again invoke LLM 
        #         async with httpx.AsyncClient(timeout= 10.0) as client:
        #             try:
        #                 to_llm_context = {"messages": messages, "tools": available_tools}
        #                 llm_response = await client.post(url = GEMMA3N_URI, headers= headers, json= to_llm_context)
        #                 llm_response.raise_for_status()            
        #             except httpx.HTTPStatusError as exc:
        #                 return f"Error Response {exc.response.status_code} while requesting {exc.request.url}"
        #             except httpx.RequestError as exc:
        #                 return f"Error whild requesting {exc.request.url}"
        #             except Exception as exc:
        #                 return f"Unexpected Error when Accesssing LLM API: \n {exc}"
                
        #         final_text.append(llm_response['message'].get("content"))


        pass 
    