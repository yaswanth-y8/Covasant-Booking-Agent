from fastapi import FastAPI, HTTPException, Body, Path, Header, Request as FastAPIRequest
from pydantic import BaseModel
from typing import Any, Optional, List, Dict
import importlib
import os
from dotenv import load_dotenv
import asyncio
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.agents.run_config import RunConfig
from google.genai.types import Content

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=DOTENV_PATH, override=True)
print(f".env loaded from {DOTENV_PATH}" if os.path.exists(DOTENV_PATH) else f"Warning: .env not found at {DOTENV_PATH}")

app = FastAPI(title="Booking Agent")

AGENT_MODULE_PATHS = {
    "movie_booking_agent": "Movie_ticket_booking_agent.agent",
    "bus_booking_agent": "Bus_ticket_booking_agent.agent",
}
loaded_agents: Dict[str, Agent] = {}

try:
    session_service = InMemorySessionService()
    print(f"Session service initialized: {session_service.__class__.__name__}")
except NameError:
    print("ERROR: InMemorySessionService not imported correctly.")
    raise
except Exception as e_sess_init:
    print(f"ERROR: Could not instantiate InMemorySessionService: {e_sess_init}")
    raise

API_DEFAULT_USER_ID = "user_1"

def get_agent_instance(agent_name: str) -> Agent:
    if agent_name in loaded_agents:
        return loaded_agents[agent_name]
    module_path = AGENT_MODULE_PATHS.get(agent_name)
    if not module_path:
        raise ValueError(f"Agent '{agent_name}' not configured in AGENT_MODULE_PATHS.")
    if agent_name not in AGENT_MODULE_PATHS:
        raise ValueError(f"Agent name '{agent_name}' is not a valid configured agent.")
    try:
        agent_module = importlib.import_module(module_path)
        if hasattr(agent_module, "root_agent") and isinstance(agent_module.root_agent, Agent):
            loaded_agents[agent_name] = agent_module.root_agent
            return agent_module.root_agent
        else:
            raise AttributeError(f"'root_agent' not found or not an Agent instance in module '{module_path}' for agent '{agent_name}'.")
    except ImportError:
        raise ImportError(f"Could not import agent module: '{module_path}' for agent '{agent_name}'.")

class StartAgentRequest(BaseModel):
    agent_name: str

class StartAgentResponse(BaseModel):
    agent_name: str
    session_id: str
    user_id: str
    query_endpoint: str

class ConverseRequest(BaseModel):
    query: str

class ConverseResponse(BaseModel):
    full_log: List[str]
    session_id: str
    user_id: str

async def _run_agent_turn_for_session(
    agent: Agent,
    session_id: str,
    user_id: str, 
    app_name: str, 
    query_text: str
) -> Dict[str, Any]:
    runner = Runner(
        app_name=app_name, agent=agent, session_service=session_service
    )
    print(f"Runner created for app: {app_name}. Using session_id: {session_id}, user_id: {user_id}")
    user_input_content = Content(parts=[{"text": query_text}], role="user")
    
    _final_agent_utterance_for_log: Optional[str] = None
    full_interaction_log = [f"User: {query_text}"]
    error_message_from_event: Optional[str] = None

    try:
        current_run_config = RunConfig()
    except NameError:
        return {
            "full_log": full_interaction_log + ["Server Error: RunConfig not configured."],
            "session_id": session_id,
            "user_id": user_id,
            "error_for_internal_processing": "Server configuration error: RunConfig missing."
        }
    
    async for event_obj in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=user_input_content, run_config=current_run_config
    ):
        try:
            event_data = event_obj.model_dump(exclude_none=True)
        except AttributeError:
            event_data = event_obj.dict(exclude_none=True)
        
        print(f"Event Data: {event_data}")
        content = event_data.get("content")
        if content and isinstance(content, dict):
            parts = content.get("parts")
            role = content.get("role")
            if role == "model" and parts and isinstance(parts, list):
                for part_data in parts:
                    if isinstance(part_data, dict):
                        if "text" in part_data and part_data["text"]:
                            _final_agent_utterance_for_log = part_data["text"]
                            full_interaction_log.append(f"Agent: {_final_agent_utterance_for_log}")
                        elif "function_call" in part_data:
                            fc = part_data["function_call"]
                            full_interaction_log.append(f"Tool Call Request: {fc.get('name')}({fc.get('args')})")
            elif (role == "tool" or role == "function") and parts and isinstance(parts, list):
                for part_data in parts:
                    if isinstance(part_data, dict) and "function_response" in part_data:
                        fr = part_data["function_response"]
                        full_interaction_log.append(f"Tool Result for {fr.get('name')}: (data processed)")
        
        if hasattr(event_obj, "error_details") and event_obj.error_details: 
            error_message_from_event = str(event_obj.error_details)
        elif event_data.get("error_message") or event_data.get("error"): 
            error_message_from_event = str(event_data.get("error_message") or event_data.get("error"))
        
        if error_message_from_event:
            full_interaction_log.append(f"Error Event: {error_message_from_event}")
            break
            
    return {
        "full_log": full_interaction_log,
        "session_id": session_id,
        "user_id": user_id,
        "error_for_internal_processing": error_message_from_event
    }

@app.post("/start_agent_interaction/", response_model=StartAgentResponse)
async def start_agent_interaction(fastapi_request: FastAPIRequest, request: StartAgentRequest = Body(...)):
    agent_name = request.agent_name
    user_id = API_DEFAULT_USER_ID
    print(f"Request to start session for agent='{agent_name}', user='{user_id}'")
    try:
        _agent = get_agent_instance(agent_name)
        print(f"Creating new session: user_id='{user_id}', app_name='{agent_name}'")
        created_session_obj: Optional[Session] = None
        if hasattr(session_service, "create_session"):
            if asyncio.iscoroutinefunction(session_service.create_session):
                created_session_obj = await session_service.create_session(
                    app_name=agent_name, user_id=user_id
                )
            else:
                created_session_obj = session_service.create_session(
                    app_name=agent_name, user_id=user_id
                )
            if not created_session_obj or not created_session_obj.id:
                raise RuntimeError(f"Session service 'create_session' did not return a valid session object with an ID.")
            current_session_id = created_session_obj.id
            print(f"New session '{current_session_id}' created by InMemorySessionService.")
        else:
            raise NotImplementedError("InMemorySessionService lacks 'create_session'.")
        
        query_endpoint_url_path = fastapi_request.url_for(
            'agent_query_turn_handler',
            agent_name_in_path=agent_name
        )
        
        return StartAgentResponse(
            agent_name=agent_name,
            session_id=current_session_id,
            user_id=user_id,
            query_endpoint=str(query_endpoint_url_path)
        )

    except (ValueError, ImportError, AttributeError, NotImplementedError) as e:
        print(f"Error starting session for agent '{agent_name}': {str(e)}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Could not start session: {str(e)}")
    except Exception as e:
        print(f"Unexpected error starting session: {str(e)}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/{agent_name_in_path}/query", response_model=ConverseResponse, name="agent_query_turn_handler")
async def agent_query_turn_handler(
    agent_name_in_path: str = Path(..., description="The name of the agent (from URL path)"),
    x_session_id: str = Header(..., alias="X-Session-Id", description="The active session ID (from header)"),
    request: ConverseRequest = Body(...)
):
    query = request.query
    agent_name = agent_name_in_path
    session_id = x_session_id
    user_id = API_DEFAULT_USER_ID

    print(f"Converse turn: agent='{agent_name}', session='{session_id}', query='{query}', user='{user_id}'")
    
    try:
        loaded_session: Optional[Session] = None
        if hasattr(session_service, "get_session"):
            if asyncio.iscoroutinefunction(session_service.get_session):
                loaded_session = await session_service.get_session(
                    session_id=session_id, user_id=user_id, app_name=agent_name
                )
            else:
                loaded_session = session_service.get_session(
                    session_id=session_id, user_id=user_id, app_name=agent_name
                )
        
        if not loaded_session:
            raise ValueError(f"Session ID '{session_id}' (from header) not found for user '{user_id}' and agent '{agent_name}'. Please start a new session via /start_agent_interaction/.")

        agent_instance = get_agent_instance(agent_name)
        print(f"Agent '{agent_name}' retrieved for session '{session_id}'.")

        turn_result_dict = await _run_agent_turn_for_session(
            agent=agent_instance,
            session_id=session_id,
            user_id=user_id,
            app_name=agent_name,
            query_text=query
        )
        return ConverseResponse(
            full_log=turn_result_dict["full_log"],
            session_id=turn_result_dict["session_id"],
            user_id=turn_result_dict["user_id"],
            error=turn_result_dict.get("error_for_internal_processing")
        )

    except ValueError as e:
        print(f"ValueError processing query for session '{session_id}': {str(e)}")
        import traceback; traceback.print_exc()
        return ConverseResponse(
            full_log=[f"User: {query}", f"Error: {str(e)}"], 
            session_id=session_id, 
            user_id=user_id, 
            error=str(e)
        )
    except Exception as e:
        print(f"Unexpected error processing query: {str(e)}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Booking Agent...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
