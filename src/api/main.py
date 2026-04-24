from fastapi import FastAPI
from src.api.routes import health, memory, agent

app = FastAPI(
    title="HMS Core API",
    description="Hybrid Memory System - Retrieval Layer",
    version="1.0.0"
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(memory.router, prefix="/api/memory", tags=["Memory"])
app.include_router(agent.router, prefix="/api/agent", tags=["Agent"])


@app.get("/")
async def root():
    return {"message": "HMS Core API", "version": "1.0.0"}