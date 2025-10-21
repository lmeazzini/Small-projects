from fastapi import FastAPI
from datetime import datetime
import uvicorn

app = FastAPI(
    title="Small Projects API",
    description="API for Data Science Small Projects",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Small Projects API",
        "version": "1.0.0",
        "endpoints": ["/", "/init", "/health"]
    }

@app.get("/init")
async def init():
    """
    Initialize endpoint
    Returns system initialization information and status
    """
    return {
        "status": "initialized",
        "message": "System initialized successfully",
        "timestamp": datetime.utcnow().isoformat(),
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

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
