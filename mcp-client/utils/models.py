import os
from typing import Literal, Optional
from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import InferenceClient 
from google.genai import Client


HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN", "False") 
GOOGLE_GEMINI_API_TOKEN = os.environ.get("GOOGLE_GEMINI_API_TOKEN", "False")

class Chat:

    def __init__(self, model: Literal["gpt-oss-20b", "gemini"]) -> None:

        self.model: str = model
        self.client: Optional[None] =  None
    
    def prepare_model(self):
        """Preapring Model from providers for processing User query.
        """
        # openai model from huggingface
        if self.model == "gpt-oss-20b" and HUGGINGFACE_API_TOKEN != "False":
            self.client = InferenceClient(
                model= f"openai/{self.model}",
                api_key= HUGGINGFACE_API_TOKEN, 
                provider= "auto" 
            )
        elif self.model == "gemini" and GOOGLE_GEMINI_API_TOKEN != "False":
            self.client = Client(
             api_key= GOOGLE_GEMINI_API_TOKEN   
            )
        else:
            raise NotImplementedError(self.model)

        return self.client