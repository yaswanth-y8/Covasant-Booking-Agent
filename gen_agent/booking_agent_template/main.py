# booking_agent_template/main.py

from fastapi import Request, HTTPException, Response, Header
from pydantic import BaseModel
from typing import Dict, List, Annotated # Annotated is important for header injection

from booking_agent_template import app
from . import agent as agent_module

class AgentConfigRequest(BaseModel):
    model_name: str = "gemini-1.5-flash-latest"
    agent_name: str
    agent_description: str
    agent_instruction: str

class QueryRequest(BaseModel):
    query: str

class AgentSessionInfo(BaseModel):
    agent_name: str
    session_id: str
    description: str

class SessionCreationResponse(BaseModel):
    message: str
    session_id: str
    agent_details: AgentSessionInfo
    interaction_endpoint_example: str


@app.get("/")
async def root_info():
    return {
        "message": "Agent Session API.",
        "endpoints": {
            "initiate_session": "POST /startagent",
            "list_sessions": "GET /agents",
            "query_session": "POST /{agent_name}/query (Requires X-Session-ID header)"
        }
    }

@app.post("/startagent", response_model=SessionCreationResponse, status_code=201)
async def initiate_agent_session(config_request: AgentConfigRequest, response: Response, http_request: Request): # FastAPI injects 'response'
    try:
        new_agent_instance = agent_module.create_new_agent_session_instance(
            model_name=config_request.model_name,
            agent_name=config_request.agent_name,
            agent_description=config_request.agent_description,
            agent_instruction=config_request.agent_instruction
        )
    except ValueError as ve:
        raise HTTPException(status_code=409, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initiate agent session: {str(e)}")

    session_id = new_agent_instance.session_id
    
    response.headers["X-Session-ID"] = session_id 
    
    agent_details = new_agent_instance.get_session_details()

    return SessionCreationResponse(
        message=f"Agent session initiated successfully for '{agent_details['agent_name']}'.",
        session_id=session_id,
        agent_details=AgentSessionInfo(**agent_details),
        interaction_endpoint_example=f"{http_request.base_url}{agent_details['agent_name']}/query  (with X-Session-ID: {session_id} in header)"
    )

@app.get("/agents", response_model=List[AgentSessionInfo])
async def list_active_sessions():
    active_sessions_details = agent_module.list_all_active_session_details()
    return [AgentSessionInfo(**details) for details in active_sessions_details]

@app.post("/{agent_name}/query")
async def query_agent_session(
    query_request: QueryRequest,
    agent_name:str,
    x_session_id: Annotated[str | None, Header(alias="X-Session-ID", convert_underscores=False)] = None
):  
    if not x_session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header is required.")

    agent_instance = agent_module.get_agent_by_session_id(x_session_id)
    
    if not agent_instance:
        raise HTTPException(status_code=404, detail=f"No active session found for X-Session-ID: {x_session_id}")

    if not query_request.query:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body.")

    try:
        agent_response = await agent_instance.call_agent_async(query_request.query)
        return {"user_query": query_request.query, "agent_response": agent_response, "session_id": x_session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during agent interaction: {str(e)}")
