from fastapi import APIRouter, HTTPException, Depends
from loguru import logger
from src.api.schemas import AgentRequest, AgentResponse
from src.agent.chat_agent import ChatAgent

router = APIRouter()

# Dependency to get or create agent
def get_agent():
    return ChatAgent()

@router.post("/ask", response_model=AgentResponse)
async def ask_agent(request: AgentRequest, agent: ChatAgent = Depends(get_agent)):
    """Ask the Hermes Memory System a question. Returns a personalized answer."""
    logger.info(f"Agent /ask called by {request.actor_id} for query: {request.query}")
    try:
        response = await agent.ask(query=request.query, actor_id=request.actor_id)
        return response
    except Exception as e:
        logger.error(f"Error in agent /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
