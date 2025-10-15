import os 
import time
import json
import requests
from pathlib import Path
from typing import List, Dict, Any
# import false
import pickle
from openai import max_retries
from sentence_transformers import SentenceTransformer

#langchain libs
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
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
        for attempt in range(max_retries):
            try:
                self._throttle()
                return self.llm.invoke(messages)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise e
                
# web search tool
@tool
def search_web(query: str) -> str:
    """
        Search the web for a query using DuckDuckGo instant Answer API and return a summary.

        Args:
            query (str): The search query.
        
        Returns:
            str: A summary of the search results.
    """
    # Implements the web search logic here
    print(f"Searching the web for: {query}")

    url = "https://api.duckduckgo.com/"

    params = {
        "q": query,
        "format": "json",
        "no_redirect": 1,
        "skip_disambig": 1,
        "t": "open_agent"
    }

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    summary = data.get("AbstractText") or ""

    if not summary:
        rel = data.get("RelatedTopics") or []
        if rel and isinstance(rel, list):
            first = rel[0]
            if isinstance(first, dict):
                summary = first.get("Text", "")
                summary  = summary[:500] + "(See more at:" + first.get("FirstUrl", "") + " )"

    return summary or "No relevant information found."



# 4. Agent loop with tools
class AgentLoop:
    def __init__(self, client: GeminiClient, tools: List[Any], max_steps: int = 5):
        self.client = client
        self.tools = tools 
        self.llm_tools = client.llm.bind_tools(tools)


    def run (self, user_message: str) -> str:
        """Run the agent loop with the user message."""

        messages: List[Any] = [
            SystemMessage(content=" You are a helpful agent, you may call tools when needed, if you call at tool output a json specifying tool name and arguments, otherwise output a final answer."),
            HumanMessage(content=user_message)
        ]
        
        seen = set() # to avoid repeating tool calls
        for _ in range(self.max_steps):
            ai: AIMessage = self.llm_tools.invoke(messages)
            if ai.tool_calls:

                tc = ai.tool_calls[0]  # only one tool call at a time
                tname, targs, tid = tc["name"], tc["args"], tc["id"]
                key = (tname, json.dumps(targs, sort_keys=True))

                if key in seen:
                    messages.append(ai)
                    messages.append(ToolMessage(tool_call_id = tid, content = json.dumps({"output": "Repeated tool call avoided"})))
                    continue    
                
                seen.add(key)

                tool_map = {t.name: t for t in self.tools}
                try:
                    abs = tool_map[tname].invoke(targs)
                except Exception as e:
                    abs = f"Error invoking tool {tname}: {str(e)}"

                messages.append(ai)
                messages.append(ToolMessage(tool_call_id = tid, content = json.dumps({"output": abs})))
                continue
            
            # No tool calls, final answer
            return ai.content
        
        return "I couldn't complete the task within the allowed steps."


# 2. Minimal Chain function (prompt -> response)
def minimal_chain(client: GeminiClient, question: str) -> str:

    msgs = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=question)
    ]

    response = client.invoke_with_retry(msgs)
    return response.content

# Run
if __name__ == "__main__":

    client = GeminiClient()

    question = "What is the population of the capital of France"
    answer = minimal_chain(client, question)

    print(f"Q: {question}\nA: {answer}")