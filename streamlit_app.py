# streamlit_app.py
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Import functions/classes from our logic file
from chatbot_logic import initialize_components, StreamlitCallbackHandler

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Mental Health Support Chatbot", layout="wide")
st.title("🧠 GenAI Mental Health Assistant")
st.caption("This chatbot offers supportive conversation and can search the web for general info. It does not provide medical advice.")

# --- Initialization ---
# Initialize the agent executor (cached)
agent_executor = initialize_components()

# Initialize chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "callback_handler" not in st.session_state:
    st.session_state.callback_handler = StreamlitCallbackHandler()

# --- Display Chat History ---
# Iterate through the existing chat history and display messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            # Check if there were intermediate search results associated with this AIMessage
            # The 'additional_kwargs' might contain info if return_intermediate_steps=True was used effectively
            # However, using the callback is more direct for this request.
            if "search_results" in message.additional_kwargs and message.additional_kwargs["search_results"]:
                 with st.expander("🔎 Show Search Results", expanded=False):
                    st.info(f"I searched the web and found this:\n\n---\n{message.additional_kwargs['search_results']}")
            st.markdown(message.content)


# --- Handle User Input ---
user_input = st.chat_input("How can I help you today?")

if user_input and agent_executor:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Add user message to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Prepare agent input
    agent_input = {
        "input": user_input,
        "chat_history": st.session_state.chat_history # Pass current history
    }

    # Use a placeholder for the assistant's response stream
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        search_results_md = "" # To store search results formatted for display

        # Create a new callback handler instance for this specific invocation
        # Or reuse the session state one, ensuring it's cleared properly
        current_callback_handler = st.session_state.callback_handler # Use the session state handler

        # Invoke the agent using stream for better responsiveness and callbacks
        try:
            # response = agent_executor.invoke(agent_input, config={"callbacks": [current_callback_handler]})
            # bot_response_content = response.get('output', "Sorry, I encountered an issue.")

            # Using stream for potentially better UX and capturing intermediate thoughts/tool calls
            assistant_response_chunks = []
            intermediate_steps_display = st.empty() # Placeholder for search results

            for chunk in agent_executor.stream(agent_input, config={"callbacks": [current_callback_handler]}):
                # Render intermediate steps (like search results) if the callback captured them
                search_results = current_callback_handler.get_search_results()
                if search_results:
                    search_results_md = f"🔎 **Searching...**\n\n---\n*Found:* {search_results}\n---"
                    intermediate_steps_display.info(search_results_md) # Display search results *above* final answer stream

                # Process different types of chunks from the stream
                if "actions" in chunk:
                    for action in chunk["actions"]:
                         # Optional: Could display a thinking indicator here
                         # message_placeholder.markdown(f"Thinking... using {action.tool}...")
                         pass # Action details are handled by callbacks/agent execution

                elif "steps" in chunk:
                    # Tool results might appear here too, handled by callback ideally
                    pass

                elif "messages" in chunk:
                    # This usually contains the final response chunks
                    for message in chunk["messages"]:
                        if isinstance(message, AIMessage):
                             assistant_response_chunks.append(message.content)
                             # Stream the response chunk by chunk to the placeholder
                             full_response = "".join(assistant_response_chunks)
                             message_placeholder.markdown(full_response + "▌") # Simulate typing cursor

            # Final update to the message placeholder
            message_placeholder.markdown(full_response)
            bot_response_content = full_response

        except Exception as e:
            st.error(f"An error occurred: {e}")
            bot_response_content = "Sorry, I encountered an error while processing your request."

        # --- Store and Display Final Response ---

        # Add assistant response to chat history
        # Store search results with the message if they were displayed
        ai_message = AIMessage(
            content=bot_response_content,
            additional_kwargs={"search_results": search_results_md if search_results_md else None} # Store formatted results
            )
        st.session_state.chat_history.append(ai_message)

        # Optional: Limit chat history size
        MAX_HISTORY_LEN = 10 # Keep last 5 pairs
        if len(st.session_state.chat_history) > MAX_HISTORY_LEN:
            st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY_LEN:]

        # Rerun is implicitly handled by Streamlit after input processing,
        # which will redraw the chat history including the new messages.

elif not agent_executor:
    st.warning("Chatbot initialization failed. Please check configuration and logs.")

# Add a footer or additional info if needed
# st.sidebar.info("...")