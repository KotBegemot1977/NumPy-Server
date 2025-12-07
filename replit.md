# NumPy FastAPI Server

## Overview
A simple FastAPI server with numpy for numerical operations. The server provides endpoints for statistical calculations, random number generation, and matrix operations.

## Project Structure
- `main.py` - Main FastAPI application with all endpoints

## Running the Server
The server runs on port 5000 using uvicorn.

## API Endpoints
- `GET /` - Welcome message
- `GET /health` - Health check with numpy version
- `POST /stats` - Calculate statistics (mean, std, min, max, sum) for an array
- `POST /random` - Generate random numbers
- `POST /matrix/transpose` - Transpose a matrix
- `POST /matrix/multiply` - Multiply two matrices

## Documentation
Access the interactive API docs at `/docs` (Swagger UI).

## Dependencies
- FastAPI
- Uvicorn
- NumPy
