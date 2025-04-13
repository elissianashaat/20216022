# Dragify-AI-Server

Dragify-AI-Server is a backend service for AI-driven automation, providing API endpoints for agents, vision-related tasks, and LLM-based AI models. The server is built using FastAPI and integrates with various AI services.

## Features

- Modular route structure for agents, vision processing, and AI models
- Authentication system with MongoDB
- Database support with PostgreSQL for user-mimicked databases
- Google Sheets API integration
- AI-powered menu extraction using Google Cloud Vision API
- AI-generated charts and plots from Excel files
- RAG-based chatbot microservice with LangChain and ChromaDB

## Project Structure

```
Dragify-AI-Server/
├── app/
│   ├── main.py                # Main entry point
│   ├── routes/                # API route files
│   │   ├── agents/            # Routes related to agents
│   │   │   ├── route1.py
│   │   │   ├── route2.py
│   │   ├── vision/            # Routes related to vision
│   │   │   ├── route1.py
│   │   │   ├── route2.py
│   │   ├── LLM/               # Routes related to AI Models
│   │   │   ├── route1.py
│   │   │   ├── route2.py
│   │   ├── users.py           # Routes related to users
│   │   ├── auth.py            # Authentication routes
│   ├── models/                # Database models (if using SQLAlchemy)
│   ├── schemas/               # Pydantic schemas
│   ├── dependencies/          # Shared dependencies
│   ├── services/              # Business logic
│   ├── config.py              # Configuration settings
├── tests/                     # Unit and integration tests
├── requirements.txt           # Dependencies
├── .env                       # Environment variables
├── .gitignore
```

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL
- MongoDB
- Google Cloud credentials (for Vision API)

### Setup

```bash
# Clone the repository
git clone https://github.com/<username>/Dragify-AI-Server.git
cd Dragify-AI-Server

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate` and on git bash use source .venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file with the required environment variables, for example:

```
DATABASE_URL=postgresql://user:password@localhost/dbname
MONGO_URI=mongodb://localhost:27017
google_application_credentials=path/to/google_credentials.json
```

## Running the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Alternatively, run this command from the root directory if you face any problems with routing/imports 
```bash
python -m main.py
```

## API Documentation

FastAPI automatically generates interactive API documentation:

- OpenAPI Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Redoc Docs: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Testing

If you create any unit/integration tests, apply them by running:
```bash
pytest tests/
```

## Deployment

Use Docker for deployment:

```bash
docker build -t dragify-ai-server .
docker run -p 8000:8000 dragify-ai-server
```

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit changes (`git commit -m 'feat: Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a pull request

## License

MIT License

## Contact

For inquiries, reach out to [itsmadatef@gmail.com].

