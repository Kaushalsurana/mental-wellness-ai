import os
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_community.utilities import GoogleSerperAPIWrapper # Import Serper wrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Ensure environment variables are set by the user in a .env file
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
# Note: AzureChatOpenAI might require a deployment name, even if not explicitly in the .env
# If issues arise, you might need to specify it directly or add it back to .env
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4") # Defaulting to gpt-4, adjust if needed

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# --- Initialization ---
def initialize_components():
    """Initializes the LLM, tools, and agent."""

    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, SERPER_API_KEY]):
         print("Error: Required environment variables are missing.")
         print("Please create a .env file based on .env.example and fill in your API keys.")
         return None, None, None # Indicate failure

    # Initialize Azure Chat LLM
    # Note: Adjust deployment_name if needed based on your Azure setup
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME, # This might be necessary
        temperature=0.7, # Adjust creativity/factuality
        max_tokens=1000
    )

    # Initialize Tools (Web Search using Serper)
    search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
    # Define the tool for the agent. Ensure the description is clear for the agent's understanding.
    tools = [
        Tool(
            name="web_search",
            func=search.run,
            description="Useful for when you need to answer questions about current events, general knowledge, or look up information you don't know. Input should be a search query."
        )
    ]

    # Define the prompt template for the agent
    # Customize this prompt to guide the chatbot's behavior for mental health support
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful and empathetic mental health assistant. You can provide supportive conversation and access web search for general information, but you must never provide medical advice. If asked for medical advice, politely decline and suggest consulting a healthcare professional. Always prioritize user safety and well-being."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Create the Agent Executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # Set verbose=True for debugging

    return llm, tools, agent_executor

# --- Main Interaction Loop ---
def main():
    print("Initializing Mental Health Chatbot...")
    llm, tools, agent_executor = initialize_components()

    if not agent_executor:
        return # Exit if initialization failed

    print("Chatbot Initialized. How can I help you today? (Type 'quit' to exit)")
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Exiting chatbot. Take care!")
            break

        # Invoke the agent
        try:
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            bot_response = response['output']
            print(f"Assistant: {bot_response}")

            # Update chat history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=bot_response))

            # Optional: Limit chat history size to prevent excessive token usage
            if len(chat_history) > 10: # Keep last 5 pairs of messages
                 chat_history = chat_history[-10:]

        except Exception as e:
            print(f"An error occurred: {e}")
            # Optionally add more robust error handling

if __name__ == "__main__":
    main()
