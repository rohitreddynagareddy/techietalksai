# Development Guidelines

## Commands
- **Run Streamlit App**: `streamlit run app-st.py`
- **Run Basic App**: `python app.py`
- **Docker Build**: `docker build -t ai-chatbot .`
- **Docker Compose**: `docker-compose up`
- **Run with Docker**: `docker run -p 8501:8501 --env-file .env ai-chatbot`
- **Stop Docker Compose**: `docker-compose down`

## Code Style Guidelines
- **Imports**: Group imports by standard library, third-party, and local modules with a blank line between groups
- **Typing**: Use type hints for function parameters and return values
- **Naming**: 
  - Use snake_case for variables and functions
  - Use PascalCase for classes
  - Use ALL_CAPS for constants
- **Error Handling**: Use try/except blocks with specific exception types
- **Documentation**: Use docstrings for functions and classes following Google style
- **Function Design**: Keep functions focused on a single responsibility
- **Comments**: Add comments for complex logic, but prefer readable code
- **Spacing**: Use 4 spaces for indentation
- **Line Length**: Limit lines to 88 characters
- **Vector DB**: Use LanceDb for local development, PgVector for production

## API Keys
Store API keys in .env file, never hardcode or commit them to the repository. Required keys: OPENAI_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY