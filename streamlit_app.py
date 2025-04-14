import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- Environment Variable Loading ---
# Load environment variables from .env file only if it exists
# This is helpful for local development. In deployment, variables might be set differently.
if os.path.exists(".env"):
    load_dotenv()
else:
    # In a deployed environment (like Streamlit Cloud), secrets are often managed differently.
    # You might use st.secrets here if deploying on Streamlit Cloud.
    # For now, we'll rely on environment variables being set system-wide or via .env
    print("Info: .env file not found. Relying on system environment variables.")


# --- Configuration & Initialization (Similar to app.py but adapted for Streamlit) ---

# Use Streamlit secrets if available (preferred for deployment) or fallback to os.getenv
AZURE_OPENAI_API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY"))
AZURE_OPENAI_ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT"))
AZURE_OPENAI_API_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION", os.getenv("AZURE_OPENAI_API_VERSION"))
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", os.getenv("SERPER_API_KEY"))

# --- Initialization Function ---
# Use Streamlit's caching to initialize components only once
@st.cache_resource(show_spinner="Initializing chatbot components...")
def initialize_components():
    """Initializes the LLM, tools, and agent executor."""

    # Check for required environment variables/secrets
    required_vars = {
        "Azure OpenAI API Key": AZURE_OPENAI_API_KEY,
        "Azure OpenAI Endpoint": AZURE_OPENAI_ENDPOINT,
        "Azure OpenAI API Version": AZURE_OPENAI_API_VERSION,
        "Serper API Key": SERPER_API_KEY
    }
    missing_vars = [name for name, value in required_vars.items() if not value]

    if missing_vars:
        st.error(f"Missing required configuration: {', '.join(missing_vars)}. Please set them in your environment or Streamlit secrets.")
        st.stop() # Stop execution if configuration is missing

    try:
        # Initialize Azure Chat LLM
        llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            model_name="gpt-4o", # Specify the model name directly
            temperature=0.7,
            max_tokens=1000,
            streaming=True # Enable streaming for better UX (optional but recommended)
        )

        # Initialize Tools (Web Search using Serper)
        search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
        tools = [
            Tool(
                name="web_search",
                func=search.run,
                description="Useful for when you need to answer questions about current events, general knowledge, or look up information you don't know. Input should be a search query."
            )
        ]

        # Define the prompt template for the agent
        # Added specific instruction about tool usage indication
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful and empathetic mental health assistant. You can provide supportive conversation and access web search for general information, but you must **never** provide medical advice or diagnosis. If asked for medical advice, politely decline and strongly suggest consulting a qualified healthcare professional. Always prioritize user safety and well-being. If you use the web_search tool, briefly mention that you are looking up information before providing the final answer."),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # Create the agent and executor
        agent = create_tool_calling_agent(llm, tools, prompt)
        # Setting return_intermediate_steps=True allows us to see the agent's thought process,
        # including tool usage, which helps in showing the "processing" state.
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

        return agent_executor

    except Exception as e:
        st.error(f"Error initializing components: {e}")
        st.stop()

# --- Streamlit UI ---

st.set_page_config(page_title="Mental Health Assistant", page_icon="🧠")
st.title("🧠 GenAI Mental Health Assistant")
st.caption("This is an AI assistant for supportive conversation. It is not a replacement for professional mental health care.")
st.warning("⚠️ **Disclaimer:** This chatbot does not provide medical advice. If you are in crisis or need medical help, please contact a healthcare professional or emergency services immediately.")

# Initialize agent executor
agent_executor = initialize_components()

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How are you feeling today? Remember, I'm here to listen, but I cannot provide medical advice."}
    ]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare chat history for the agent
    # Convert session state messages to LangChain's expected format
    chat_history = []
    for msg in st.session_state.messages[:-1]: # Exclude the current user prompt
         if msg["role"] == "user":
             chat_history.append(HumanMessage(content=msg["content"]))
         elif msg["role"] == "assistant":
             chat_history.append(AIMessage(content=msg["content"]))

    # Display "thinking" indicator and run the agent
    with st.chat_message("assistant"):
        # Use st.status to show intermediate steps if return_intermediate_steps is True
        with st.status("Assistant is thinking...", expanded=False) as status:
            try:
                st.write("Processing your request...") # Initial message inside status
                response = agent_executor.invoke({
                    "input": prompt,
                    "chat_history": chat_history
                })
                output = response.get('output', "Sorry, I encountered an issue.")

                # Check if intermediate steps are available and if search was used
                if 'intermediate_steps' in response and response['intermediate_steps']:
                    status.update(label="Analyzing information...", state="running", expanded=False)
                    # You could potentially parse intermediate_steps here for more detail
                    # For example, check if 'web_search' tool was called
                    tool_calls = [step[0].tool for step in response['intermediate_steps'] if hasattr(step[0], 'tool')]
                    if "web_search" in tool_calls:
                         # The LLM was instructed to mention this in its response,
                         # but we can also update the status here for extra clarity.
                         st.write("Searching the web for relevant information...")


                # Update the status upon completion before displaying the final message
                status.update(label="Task complete!", state="complete", expanded=False)

                # Display the final response
                st.markdown(output)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": output})

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, I ran into a problem processing your request."})
                status.update(label="Error", state="error", expanded=True)
                st.exception(e) # Show traceback within the status context


    # Optional: Limit chat history size (in session state)
    # Keep the system message + last N pairs (e.g., last 5 pairs = 10 messages)
    max_history_length = 11 # 1 system-like message + 10 user/assistant messages
    if len(st.session_state.messages) > max_history_length:
        # Keep the first message (initial assistant greeting) and the latest messages
        st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-10:]