from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI(
    title="NumPy FastAPI Server",
    description="A simple FastAPI server with numpy for numerical operations",
    version="1.0.0"
)


class ArrayInput(BaseModel):
    numbers: list[float]


class ArrayStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    sum: float
    length: int


class MatrixInput(BaseModel):
    matrix: list[list[float]]


@app.get("/")
async def root():
    return {"message": "Welcome to the NumPy FastAPI Server", "docs": "/docs"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "numpy_version": np.__version__}


@app.post("/stats", response_model=ArrayStats)
async def calculate_stats(data: ArrayInput):
    arr = np.array(data.numbers)
    return ArrayStats(
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        sum=float(np.sum(arr)),
        length=len(arr)
    )


@app.post("/random")
async def generate_random(size: int = 10, min_val: float = 0, max_val: float = 100):
    arr = np.random.uniform(min_val, max_val, size)
    return {"random_array": arr.tolist()}


@app.post("/matrix/transpose")
async def transpose_matrix(data: MatrixInput):
    matrix = np.array(data.matrix)
    transposed = matrix.T
    return {"original_shape": list(matrix.shape), "transposed": transposed.tolist()}


@app.post("/matrix/multiply")
async def matrix_multiply(matrix_a: MatrixInput, matrix_b: MatrixInput):
    a = np.array(matrix_a.matrix)
    b = np.array(matrix_b.matrix)
    try:
        result = np.dot(a, b)
        return {"result": result.tolist(), "shape": list(result.shape)}
    except ValueError as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
