# streamlit_app.py
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Import functions/classes from our logic file
from chatbot_logic import initialize_components, StreamlitCallbackHandler

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Mental Health Support Chatbot", layout="wide", initial_sidebar_state="collapsed")
st.title("🧠 GenAI Mental Health Assistant")
st.caption("This chatbot offers supportive conversation and can search the web for general info. It does *not* provide medical advice.")
st.divider() # Adds a visual separator

# --- Initialization ---
# Initialize the agent executor (cached)
agent_executor = initialize_components()

# Initialize chat history and callback handler in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# Ensure callback handler is created once per session or reset appropriately
# Using cache_resource for the agent already handles re-creation if needed.
# For the callback, we might want one instance per session.
if "callback_handler" not in st.session_state:
    st.session_state.callback_handler = StreamlitCallbackHandler()

# --- Display Chat History ---
# Store search results associated with messages to redisplay correctly
def display_chat_history():
    for i, message in enumerate(st.session_state.chat_history):
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                # Check if search results were stored from the generation process
                search_results_html = message.additional_kwargs.get("search_results_html", None)
                if search_results_html:
                    # Use an expander to show the results that *led* to this message
                    with st.expander("🔎 Show Search Information Used", expanded=False):
                        # Use unsafe_allow_html=True carefully if the source is trusted (like formatted markdown/html we create)
                        st.markdown(search_results_html, unsafe_allow_html=True)
                st.markdown(message.content) # Display the main message content

display_chat_history() # Display history initially

# --- Handle User Input ---
user_input = st.chat_input("How can I help you today?")

if user_input and agent_executor:
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Add user message to history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Prepare agent input
    agent_input = {
        "input": user_input,
        # Pass only a relevant subset of history if it gets too long
        "chat_history": st.session_state.chat_history[-10:] # Pass last 5 pairs
    }

    # --- Interaction Handling with Loading and Streaming ---
    with st.chat_message("assistant"):
        # 1. Intermediate display area for search results (appears above final answer)
        intermediate_display = st.empty()
        # 2. Placeholder for the final streaming response
        message_placeholder = st.empty()
        # 3. Loading Spinner
        with st.spinner("👩‍⚕️ Assistant is thinking..."):
            full_response = ""
            search_results_content = None # To store raw search results
            search_displayed = False # Flag to display search results only once

            # Get the persistent callback handler
            current_callback_handler = st.session_state.callback_handler

            try:
                assistant_response_chunks = []
                # Use the stream method
                for chunk in agent_executor.stream(agent_input, config={"callbacks": [current_callback_handler]}):
                    # --- Check for Search Results (using Callback) ---
                    # Check if the callback handler captured new results *during* this stream
                    search_results = current_callback_handler.get_search_results() # get_search_results clears the stored results
                    if search_results and not search_displayed:
                        search_results_content = search_results # Store raw results
                        # Format for display in the intermediate area
                        search_results_html = f"""
                        <details>
                            <summary>🔎 I looked this up on the web...</summary>
                            <div style="padding: 10px; border: 1px solid #eee; border-radius: 5px; margin-top: 5px; background-color: #f9f9f9;">
                                <strong>Search Results Summary:</strong><br>
                                <em>{search_results_content}</em>
                            </div>
                        </details>
                        """
                        intermediate_display.markdown(search_results_html, unsafe_allow_html=True)
                        search_displayed = True # Mark as displayed for this turn

                    # --- Stream Final Answer Chunks ---
                    # Look for message content chunks
                    messages = chunk.get("messages", [])
                    for message in messages:
                         if isinstance(message, AIMessage) and message.content:
                             new_content = message.content
                             assistant_response_chunks.append(new_content)
                             full_response = "".join(assistant_response_chunks)
                             # Update placeholder with streaming text + typing cursor
                             message_placeholder.markdown(full_response + "▌")

                # Final update to remove the typing cursor
                message_placeholder.markdown(full_response)
                bot_response_content = full_response

            except Exception as e:
                st.error(f"An error occurred: {e}")
                bot_response_content = "Sorry, I encountered an error processing your request."
                message_placeholder.markdown(bot_response_content) # Display error in the placeholder

    # --- Store Final Response and Rerun ---
    # Store the final message along with any search results that were displayed
    formatted_search_results_for_storage = None
    if search_displayed and search_results_content:
         # Store the same HTML used for display, or just the raw content
         formatted_search_results_for_storage = f"""
            <div style="padding: 10px; border: 1px solid #eee; border-radius: 5px; margin-top: 5px; background-color: #f9f9f9;">
                <strong>Search Results Summary:</strong><br>
                <em>{search_results_content}</em>
            </div>
            """

    ai_message = AIMessage(
        content=bot_response_content,
        additional_kwargs={
            "search_results_html": formatted_search_results_for_storage
            }
    )
    st.session_state.chat_history.append(ai_message)

    # Optional: Limit chat history size (applied *after* adding the latest message)
    MAX_HISTORY_LEN = 10 # Keep last 5 pairs
    if len(st.session_state.chat_history) > MAX_HISTORY_LEN:
        st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY_LEN:]

    # Streamlit will automatically rerun after the script finishes,
    # redrawing the history including the latest user and assistant messages.
    # No explicit rerun needed here.

elif not agent_executor and user_input:
    st.warning("Chatbot initialization failed. Cannot process request. Please check configuration and logs.")

# --- Optional: Add a sidebar for info or settings ---
# st.sidebar.header("About")
# st.sidebar.info("This is a GenAI chatbot for mental health support...")