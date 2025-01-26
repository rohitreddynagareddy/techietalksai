# Techietalks AI - Smart Chat Assistant (Enhanced)

Welcome to the **Techietalks AI** repository! This project is an enhanced version of a smart AI-powered chat assistant built using **Streamlit**, **DeepSeek API**, and **Pydantic_AI**. It allows users to interact with an AI assistant that can answer questions conversationally and concisely, while also maintaining a history of the conversation for context.

This repository contains all the necessary files to set up and run the chat assistant locally or in a Docker container. Below, you'll find a detailed explanation of the project and instructions to get started.

---

## 📁 Repository Structure

Here’s a breakdown of the files in this repository:

1. **`.DS_Store`**: A macOS-specific file that stores folder attributes (e.g., icon positions). You can ignore this file.
2. **`.env`**: A file to store environment variables, such as API keys. **Important**: Add your `DEEPSEEK_API_KEY` here to authenticate with the DeepSeek API.
3. **`.gitignore`**: Specifies files and folders that Git should ignore (e.g., `.env` to avoid exposing sensitive information).
4. **`Dockerfile`**: Contains instructions to build a Docker image for the application. It sets up a Python environment, installs dependencies, and runs the Streamlit app.
5. **`README.md`**: This file! It provides an overview of the project and instructions for setup.
6. **`app.py`**: The main application file. It contains the code for the Streamlit-based chat interface and integrates with the DeepSeek API. This version also includes conversation history for better context.
7. **`docker-compose.yml`**: A configuration file to run the application using Docker Compose. It maps ports, loads environment variables, and sets up volumes.
8. **`requirements.txt`**: Lists all Python dependencies required to run the application.
9. **`sree.txt`**: A placeholder text file (likely for personal notes or testing).

---

## 🚀 How It Works

This project uses the following technologies:

- **Streamlit**: A framework for building interactive web applications with Python. It powers the chat interface.
- **DeepSeek API**: An AI model that provides conversational responses. The app authenticates with the API using an API key stored in the `.env` file.
- **Pydantic_AI**: A library that helps structure and validate AI responses.
- **Rich**: A library used for pretty-printing debug information (e.g., conversation history).

When you type a question into the chat interface, the app sends it to the DeepSeek API, processes the response, and displays it in a conversational format. It also maintains a history of the conversation, allowing the AI to provide more context-aware answers.

---

## 🛠️ Setup Instructions

Follow these steps to set up and run the project:

### Prerequisites
1. **Python 3.11**: Ensure Python is installed on your system.
2. **Docker** (optional): If you want to run the app in a container.
3. **DeepSeek API Key**: Obtain an API key from [DeepSeek](https://www.deepseek.com/).

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/schogini/techietalksai.git
   cd techietalksai/0003
   ```

2. **Set Up Environment Variables**:
   - Create a `.env` file in the project directory.
   - Add your DeepSeek API key:
     ```plaintext
     DEEPSEEK_API_KEY=your_api_key_here
     ```

3. **Install Dependencies**:
   - If running locally:
     ```bash
     pip install -r requirements.txt
     ```
   - If using Docker, skip this step.

4. **Run the Application**:
   - **Locally**:
     ```bash
     streamlit run app.py
     ```
     Open your browser and navigate to `http://localhost:8501`.
   - **Using Docker**:
     ```bash
     docker-compose up --build
     ```
     Open your browser and navigate to `http://localhost:8502`.

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

Enjoy using the **Techietalks AI Smart Chat Assistant**! If you have any questions or feedback, feel free to reach out. 😊