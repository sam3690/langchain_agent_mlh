import os 
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import false
import pickle
from sentence_transformers import SentenceTransformer

#langchain libs
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import Tool
import gradio as gr #UI/Frontend lib

#loading api key 
from dotenv import load_dotenv

load_dotenv()
# 1.Gemini Client
class GeminiClient:
    """Wrapper over the Gemini-langchain model with throttle and exponential backoff"""
    def __init__(self, model: str = "gemini-2.0-flash-lite", min_interval: float = 0.5, max_retries: int = 5):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        self.llm = ChatGoogleGenerativeAI(model=model, api_key=api_key)
        self.last_call_time = 0.0
        self.min_interval = min_interval
    

    def _throttle(self):
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call_time = time.time()
    

    def invoke_with_retry(self, messages: List[Any]) -> AIMessage:
        """Invoke the Gemini model with messages, handling throttling and retries."""
        backoff = 1.0
        for attempt in range(self.max_retries):
            try:
                self._throttle()
                return self.llm.invoke(messages)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise e
    # 2. Minimal Chain function (prompt -> response)
def minimal_chain(client: GeminiClient, question: str) -> str:
    msgs = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=question)
    ]
    response = client.invoke_with_retry(msgs)
    return response.content