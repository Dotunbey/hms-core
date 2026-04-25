import json
from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.config.settings import settings
from src.memory.retriever import HybridRetriever
from src.api.schemas import AgentResponse
from src.utils.audit import AuditLogger


class ChatAgent:
    """Agent that uses the Hermes Memory System to answer questions."""

    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.3
        )
        self.audit_logger = AuditLogger()

    async def ask(self, query: str, actor_id: str) -> AgentResponse:
        """Process a user query, retrieve memory, and generate a personalized response."""
        logger.info(f"Agent received query: {query}")
        
        # 1. Retrieve context from memory system
        memory_result = await self.retriever.search(query=query, top_k=5, include_graph=True)
        
        # 2. Extract sources and format context
        context_parts = []
        sources = set()
        
        for mem in memory_result.results:
            source = mem.metadata.get("source_id", "Unknown")
            sources.add(source)
            context_parts.append(f"[Source: {source}]\n{mem.content}\n")
            
        for entity in memory_result.entities:
            context_parts.append(f"[Entity Graph] {entity.label}: {json.dumps(entity.properties)}")
            
        for rel in memory_result.relationships:
            context_parts.append(f"[Relation] {rel.source_id} -[{rel.relation_type}]-> {rel.target_id}: {json.dumps(rel.properties)}")
            
        context_str = "\n".join(context_parts)
        
        if not context_str:
            context_str = "No relevant context found in memory."
            
        # 3. Construct prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Hermes Memory System agent. "
                       "You must answer the user's question using ONLY the provided memory context. "
                       "If the context does not contain the answer, politely state that you do not have that information in your memory. "
                       "Be helpful, clear, and professional.\n\nContext:\n{context}"),
            ("user", "{query}")
        ])
        
        # 4. Generate response
        try:
            chain = prompt | self.llm
            response = await chain.ainvoke({"context": context_str, "query": query})
            answer = response.content
        except Exception as e:
            logger.error(f"Agent failed to generate response: {e}")
            answer = "I'm sorry, I encountered an error while processing your request."
            
        # 5. Log the interaction
        source_list = list(sources)
        AuditLogger.log_event(
            actor=actor_id,
            action="AGENT_QUERY",
            data_accessed=[mem.id for mem in memory_result.results],
            approval_status="APPROVED",
            metadata={"query": query, "sources": source_list}
        )
        
        return AgentResponse(
            answer=answer,
            sources=source_list
        )
