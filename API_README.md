# Small Projects API

A FastAPI-based REST API for the Small Projects data science repository.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload
```

The server will start at `http://localhost:8000`

## API Endpoints

### GET /

Root endpoint that returns API information and available endpoints.

**Response:**
```json
{
  "message": "Welcome to Small Projects API",
  "version": "1.0.0",
  "endpoints": ["/", "/init", "/health"]
}
```

### GET /init

Initialize endpoint that returns system initialization information and status.

**Response:**
```json
{
  "status": "initialized",
  "message": "System initialized successfully",
  "timestamp": "2025-10-21T12:00:00.000000",
  "available_projects": [
    "House Pricing",
    "MNIST",
    "FIFA 2017 player position",
    "Titanic",
    "Vader Sentiment Analysis",
    "Neural Network from scratch",
    "Unsupervised Credit Fraud detection"
  ],
  "api_version": "1.0.0"
}
```

### GET /health

Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-21T12:00:00.000000"
}
```

## Interactive Documentation

Once the server is running, you can access:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
.
├── main.py                    # FastAPI application
├── requirements.txt           # Python dependencies
├── API_README.md             # API documentation
└── [Jupyter notebooks]       # Data science projects
```
