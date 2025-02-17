# AI Thai Cooking Assistant

The AI Thai Cooking Assistant is an interactive web application developed with Streamlit. It leverages the Agno library to create an AI agent that provides users with authentic Thai recipes, cooking techniques, and cultural insights. The assistant combines information from a curated knowledge base and real-time web searches to deliver comprehensive responses to user queries.

## Features

- **Authentic Recipe Knowledge Base:** Access a collection of over 50 traditional Thai recipes sourced from a curated PDF.
- **Real-Time Web Search Integration:** Utilize DuckDuckGo tools to fetch modern substitutions, historical contexts, and additional cooking tips.
- **Interactive User Interface:** Engage with a user-friendly chat interface to ask questions and receive detailed responses.
- **Streaming Responses:** Experience real-time answer generation with optional streaming capabilities.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/techietalksai
   cd 003-thai-cooking-assistant-rag-web-stream.git
   ```

2. **Set Up Environment Variables:**

   Edit a `.env.example` and rename it as `.env` after adding your API Keys

3. **Launch:**

   ```bash
   docker compose up --build
   ```

## Usage

1. **Browse:**

http://localhost:8511/


2. **Interact with the Assistant:**

   - Use the sidebar to explore example queries or input your own Thai cooking questions.
   - View responses directly in the main chat interface.

## How It Works

The AI Thai Cooking Assistant utilizes the Agno library to construct an AI agent with access to both a static knowledge base and dynamic web search tools.

- **Agent Configuration:**

  The agent is initialized with specific instructions to emulate a passionate Thai cuisine expert. It prioritizes information from the knowledge base and supplements with web searches when necessary.

  ```python
  from agno.agent import Agent
  from agno.models.openai import OpenAIChat
  from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
  from agno.vectordb.lancedb import LanceDb, SearchType
  from agno.embedder.openai import OpenAIEmbedder
  from agno.tools.duckduckgo import DuckDuckGoTools
  from textwrap import dedent
  import os

  openai_api_key = os.getenv("OPENAI_API_KEY")

  agent = Agent(
      model=OpenAIChat(id="gpt-4o-mini", api_key=openai_api_key),
      instructions=dedent(\"\"\"
          You are a passionate and knowledgeable Thai cuisine expert! 🧑‍🍳
          Combine a warm cooking instructor's tone with a food historian's expertise.

          Answer strategy:
          1. First check the recipe knowledge base for authentic information
          2. Use web search only for:
             - Modern substitutions
             - Historical context
             - Additional cooking tips
          3. Prioritize knowledge base content for recipes
          4. Clearly cite sources when using web information

          Response format:
          🌶️ Start with relevant emoji
          📖 Structure clearly:
          - Introduction/context
          - Main content (recipe/steps/explanation)
          - Pro tips & cultural insights
          - Encouraging conclusion

          For recipes include:
          📝 Ingredients with substitutions
          🔢 Numbered steps
          💡 Success tips & common mistakes

          Special features:
          - Explain Thai ingredients & alternatives
          - Share cultural traditions
          - Adapt recipes for dietary needs
          - Suggest serving pairings

          End with:
          - 'Happy cooking! ขอให้อร่อย (Enjoy your meal)!'
          - 'May your Thai cooking adventure bring joy!'
          - 'Enjoy your homemade Thai feast!'
      \"\"\"),
      knowledge=PDFUrlKnowledgeBase(
          urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
          vector_db=LanceDb(
              uri="tmp/lancedb",
              table_name="recipe_knowledge",
              search_type=SearchType.hybrid,
              embedder=OpenAIEmbedder(id="text-embedding-3-small", api_key=openai_api_key),
          ),
      ),
      tools=[DuckDuckGoTools()],
      show_tool_calls=True,
      markdown=True,
      add_references=True,
    ```

- **Knowledge Base Loading**

  The agent loads a PDF containing authentic Thai recipes into a vector database for efficient reteval

  ```python
  if agent.knowledge:
      agent.knowledge.load(  ```

- **User Interaction**

  Users input their questions through a text input field, and the agent processes these prompts to generate informative and engaging resnses

  ```python
  import streamlit as st

  prompt = st.text_input("Ask your Thai cooking question (e.g., 'How to make Pad Thai?')")

  if prompt:
      response = agent.run(prompt, stream=False)
      st.markdown(response.content  ```

## Dependencies

- [Streamlit](https://streamlit.io/)
- [Agno](https://github.com/agno-agi/agno)
- [OpenAI](https://openai.com/)

## Liense

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for dails.

## Acknowledgmets

- [Streamlit](https://streamlit.io/) for providing an intuitive framework for web applicion.
-  
