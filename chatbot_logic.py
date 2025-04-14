# chatbot_logic.py
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st # Import streamlit to use for displaying intermediate steps

# --- Configuration ---
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
AZURE_OPENAI_MODEL_NAME = "gpt-4o" # Or your preferred model

# --- Custom Callback Handler ---
class StreamlitCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to display tool results in Streamlit."""

    def __init__(self):
        self.search_results = None

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Optional: Could indicate that a tool is starting."""
        # print(f"Tool Start: {serialized.get('name')}")
        pass

    def on_tool_end(self, output, **kwargs):
        """Capture the output of the tool (search results)."""
        # Assuming the tool used is the web_search tool
        # print(f"Tool End Output: {output}") # Debugging
        self.search_results = output
        # Display results immediately using a dedicated area (requires passing st object or using session state)
        # For simplicity here, we store it and let the main app display it.

    def get_search_results(self):
        """Retrieve and clear the stored search results."""
        results = self.search_results
        self.search_results = None # Clear after retrieval
        return results

# --- Initialization Function ---
# Use Streamlit's caching to avoid re-initializing on every interaction
@st.cache_resource(show_spinner="Initializing Chatbot Components...")
def initialize_components():
    """Initializes the LLM, tools, and agent. Returns agent_executor."""

    # Check for required environment variables
    required_vars = {
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
        "SERPER_API_KEY": SERPER_API_KEY
    }
    missing_vars = [k for k, v in required_vars.items() if not v]

    if missing_vars:
        st.error(f"Required environment variables missing: {', '.join(missing_vars)}. Please check your .env file.")
        return None # Indicate failure

    # Initialize Azure Chat LLM
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            model_name=AZURE_OPENAI_MODEL_NAME,
            temperature=0.7,
            max_tokens=1000,
            streaming=True, # Enable streaming for better UX
        )
    except Exception as e:
        st.error(f"Error initializing AzureChatOpenAI: {e}")
        return None # Indicate failure

    # Initialize Tools (Web Search using Serper)
    try:
        search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
    except Exception as e:
        st.error(f"Error initializing GoogleSerperAPIWrapper: {e}. Ensure SERPER_API_KEY is correct.")
        return None

    # Define the tool for the agent
    tools = [
        Tool(
            name="web_search",
            func=search.run,
            description="Useful for when you need to answer questions about current events, general knowledge, or look up information you don't know. Input should be a search query."
        )
    ]

    # Define the prompt template for the agent
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful and empathetic mental health assistant. You can provide supportive conversation and access web search for general information, but you MUST NEVER provide medical advice. If asked for medical advice, politely decline and suggest consulting a healthcare professional. Always prioritize user safety and well-being. When using web search, briefly mention that you are looking things up."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Create the Agent Executor
    # We pass return_intermediate_steps=True to potentially access tool usage details
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True, # Keep verbose for debugging during development
        handle_parsing_errors=True, # Gracefully handle potential parsing errors
        # return_intermediate_steps=True # Useful if not using callbacks/streaming
    )

    return agent_executor