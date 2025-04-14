# GenAI Mental Health Assistant 🧠

This project is a research prototype for a Generative AI-based mental health chatbot. It utilizes Large Language Models (LLMs) via Azure OpenAI services and incorporates web search capabilities through the Serper API to provide informative and supportive conversations.

**⚠️ Disclaimer:** This is a research project and **not** a substitute for professional mental health diagnosis or treatment. It is designed for supportive conversation only and cannot provide medical advice. If you are in crisis or require medical assistance, please contact a qualified healthcare professional or emergency services immediately.

## Features

* **Empathetic Conversation:** Powered by Azure OpenAI's GPT models (specifically configured for gpt-4o in the code) to engage in supportive dialogue.
* **Web Search Integration:** Uses an AI agent (built with Langchain) equipped with a Serper API tool to look up general information or current events when needed.
* **Safety First:** The chatbot is explicitly instructed **not** to provide medical advice and to redirect users to professional help when appropriate.
* **Streamlit Interface:** Provides a user-friendly web interface for interaction.
* **Processing Indicator:** Shows visual feedback in the chat interface while the assistant is thinking or searching the web.
* **Session History:** Remembers the conversation context within a session.

## Tech Stack

* **Python:** Core programming language.
* **Langchain:** Framework for building LLM applications and agents.
* **Azure OpenAI:** Provides the core LLM capabilities (requires Azure credits/subscription).
* **Google Serper API:** Enables the web search tool (requires a Serper API key).
* **Streamlit:** Framework for building the interactive web UI.
* **python-dotenv:** For managing environment variables locally.

## Project Structure
```
genai-mental-health-chatbot/
├── .env # Stores sensitive API keys (DO NOT COMMIT)
├── .env.example # Example environment file structure
├── app.py # Original command-line application logic
├── streamlit_app.py # The Streamlit web interface application
├── requirements.txt # Python dependencies
└── README.md # This file
```

## Setup and Installation

1. **Prerequisites:**
   * Python 3.8+
   * Access to Azure OpenAI services (Endpoint, API Key, API Version)
   * A Serper API Key (https://serper.dev/)

2. **Clone the Repository:**
   ```bash
   git clone <your-repo-url>
   cd genai-mental-health-chatbot
   ```

3. **Create a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure Environment Variables:**
   * Copy the .env.example file to a new file named .env:
     ```bash
     cp .env.example .env
     ```
   * Open the .env file and fill in your actual API keys and Azure endpoint details:
     ```
     # Azure OpenAI Credentials
     AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY"
     AZURE_OPENAI_ENDPOINT="YOUR_AZURE_OPENAI_ENDPOINT"
     AZURE_OPENAI_API_VERSION="YOUR_AZURE_API_VERSION"

     # Serper API Key for Web Search
     SERPER_API_KEY="YOUR_SERPER_API_KEY"
     ```
   * **Important:** Ensure the .env file is added to your .gitignore file to prevent accidentally committing sensitive keys.

## Running the Application

### Streamlit Web Interface (Recommended)

Ensure your virtual environment is activated and the .env file is configured.

```bash
streamlit run streamlit_app.py
```

This will start the Streamlit server, and you can access the chatbot interface in your web browser (usually at http://localhost:8501).

### Command-Line Interface (For Core Logic Testing)

You can also run the basic command-line version:

```bash
python app.py
```

## Important Considerations & Ethics

* **Not a Medical Device:** Reiterate that this tool is not for diagnosis or treatment.
* **User Privacy:** Be mindful of the data being sent to the LLM API. Avoid inputting highly sensitive personal health information. Consider data handling policies if developing further.
* **Safety & Hallucinations:** LLMs can sometimes generate incorrect or inappropriate information (hallucinations). The safety prompts aim to mitigate this, but vigilance is required.
* **Bias:** LLMs can reflect biases present in their training data. Be aware of potential biases in responses.
* **Crisis Management:** This chatbot is not equipped to handle crisis situations. Users in crisis should be directed to appropriate resources.

## Future Development Ideas

* **Sentiment Analysis:** Analyze user input sentiment to tailor responses.
* **Memory:** Implement more sophisticated long-term memory solutions.
* **Specialized Tools:** Add tools for specific supportive exercises (e.g., guided breathing - with caution not to seem like medical advice).
* **User Profiles:** Allow users to personalize the experience (requires careful consideration of privacy).
* **Evaluation:** Develop metrics to evaluate the helpfulness, safety, and empathy of the chatbot's responses.
* **Fine-tuning:** Fine-tune a base model specifically for empathetic, supportive dialogue within safety constraints.
