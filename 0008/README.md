# Techietalks AI - Multi-Model Chat Assistant with RAG and Tool Integration

Welcome to the **Techietalks AI** repository! This project is a **multi-model AI-powered chat assistant** built using **Streamlit**, **Pydantic_AI**, and multiple AI models like **OpenAI**, **DeepSeek**, and **Gemini**. It allows users to interact with different AI models, switch between them, and enjoy a conversational experience. Additionally, this version includes **Retrieval-Augmented Generation (RAG)** functionality for processing PDF documents and **tool integration** for advanced query handling.

This repository contains all the necessary files to set up and run the chat assistant locally or in a Docker container. Below, you'll find a detailed explanation of the project and instructions to get started.

---

## 📁 Repository Structure

Here’s a breakdown of the files in this repository:

1. **`.DS_Store`**: A macOS-specific file that stores folder attributes (e.g., icon positions). You can ignore this file.
2. **`.env`**: A file to store environment variables, such as API keys. **Important**: Add your `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, and `GEMINI_API_KEY` here to authenticate with the respective APIs.
3. **`.gitignore`**: Specifies files and folders that Git should ignore (e.g., `.env` to avoid exposing sensitive information).
4. **`Dockerfile`**: Contains instructions to build a Docker image for the Streamlit application. It sets up a Python environment, installs dependencies, and runs the Streamlit app.
5. **`app.py`**: The main application file. It contains the code for the Streamlit-based chat interface, integrates with multiple AI models (OpenAI, DeepSeek, and Gemini), includes **RAG functionality** for processing PDF documents, and supports **tool integration** for advanced query handling.
6. **`docker-compose.yml`**: A configuration file to run the application using Docker Compose. It sets up two services: the **chatbot** (Streamlit app) and the **webserver** (NGINX for serving static content).
7. **`requirements.txt`**: Lists all Python dependencies required to run the application.
8. **`sree.txt`**: A placeholder text file (likely for personal notes or testing).
9. **`data/`**: A directory containing:
   - **`pdfs/`**: Stores uploaded PDF documents.
   - **`vector_store/`**: Stores vector embeddings of the PDF documents for RAG functionality.
10. **`web/static/index.html`**: A static HTML file served by the NGINX web server. You can customize this file to add additional content or documentation.

---

## 🚀 How It Works

This project uses the following technologies:

- **Streamlit**: A framework for building interactive web applications with Python. It powers the chat interface.
- **Pydantic_AI**: A library that helps structure and validate AI responses.
- **OpenAI API**: A powerful AI model for generating conversational responses.
- **DeepSeek API**: An AI model that provides conversational responses.
- **Gemini API**: Another AI model for generating responses.
- **Retrieval-Augmented Generation (RAG)**: A technique that combines retrieval of relevant information from documents (e.g., PDFs) with generative AI to provide context-aware answers.
- **Tool Integration**: Allows the AI to use custom tools (e.g., document retrieval) to enhance its responses.
- **NGINX**: A web server used to serve static content (e.g., HTML files).
- **Rich**: A library used for pretty-printing debug information (e.g., conversation history).

When you type a question into the chat interface, the app sends it to the selected AI model (OpenAI, DeepSeek, or Gemini), processes the response, and displays it in a conversational format. It also maintains a history of the conversation, allowing the AI to provide more context-aware answers. The **"New Chat"** button resets the conversation, clearing the history and starting fresh.

The **RAG functionality** allows users to upload PDF documents, which are processed and used to provide context-aware answers. The app uses **ChromaDB** to store vector embeddings of the documents and retrieve relevant information when answering questions.

The **tool integration** feature enables the AI to use custom tools (e.g., document retrieval) to enhance its responses, making it more powerful and versatile.

---

## 🛠️ Setup Instructions

Follow these steps to set up and run the project:

### Prerequisites
1. **Python 3.11**: Ensure Python is installed on your system.
2. **Docker**: Required to run the application in containers.
3. **API Keys**: Obtain API keys for **OpenAI**, **DeepSeek**, and **Gemini** from their respective websites.

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/schogini/techietalksai.git
   cd techietalksai/0008
   ```

2. **Set Up Environment Variables**:
   - Create a `.env` file in the project directory.
   - Add your API keys:
     ```plaintext
     OPENAI_API_KEY=your_openai_api_key_here
     DEEPSEEK_API_KEY=your_deepseek_api_key_here
     GEMINI_API_KEY=your_gemini_api_key_here
     ```

3. **Run the Application**:
   - Using Docker Compose:
     ```bash
     docker-compose up --build
     ```
   - Once the containers are running:
     - Access the **Chat Assistant** at `http://localhost:8502`.
     - Access the **Web Server** (static content) at `http://localhost:8503`.

4. **Upload PDFs**:
   - Use the sidebar in the chat interface to upload PDF documents. The app will process them and use their content to provide context-aware answers.

---

## 📄 License

This project is open-source and available under the MIT License. Feel free to use, modify, and distribute it as needed.

---

## 📧 Contact

For **AI consultancy, training, and development**, contact **Schogini Systems Private Limited** at [https://www.schogini.com](https://www.schogini.com).

---

## 👨‍💻 Author

**Sreeprakash Neelakantan**  
- Website: [https://www.schogini.com](https://www.schogini.com)  
- GitHub: [https://github.com/schogini/techietalksai.git](https://github.com/schogini/techietalksai.git)

---

Enjoy using the **Techietalks AI Multi-Model Chat Assistant with RAG and Tool Integration**! If you have any questions or feedback, feel free to reach out. 😊