from fastapi import FastAPI, HTTPException, Body, Path, Request as FastAPIRequest
from pydantic import BaseModel
from typing import  Optional, List, Dict
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

app = FastAPI(title="Dynamic Endpoint ADK Agent Service")

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

API_DEFAULT_USER_ID = "fixed_api_user_for_adk"

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
    initial_greeting_query: Optional[str] = None

class StartAgentResponse(BaseModel):
    agent_name: str
    session_id: str
    user_id: str
    query_endpoint: str
    initial_agent_utterance: Optional[str] = None
    error: Optional[str] = None

class ConverseRequest(BaseModel):
    query: str

class ConverseResponse(BaseModel):
    final_utterance: Optional[str] = None
    full_log: List[str]
    session_id: str
    user_id: str
    error: Optional[str] = None

async def _run_agent_turn_for_session(
    agent: Agent,
    session_id: str,
    user_id: str,
    app_name: str,
    query_text: str
) -> ConverseResponse:
    runner = Runner(
        app_name=app_name,
        agent=agent,
        session_service=session_service
    )
    print(f"Runner created for app: {app_name}. Using session_id: {session_id}, user_id: {user_id}")
    
    user_input_content = Content(parts=[{"text": query_text}], role="user")
    
    final_agent_utterance: Optional[str] = None
    full_interaction_log = [f"User ({user_id}): {query_text}"]
    error_message_from_event: Optional[str] = None

    try:
        current_run_config = RunConfig()
    except NameError:
        return ConverseResponse(final_utterance=None, full_log=full_interaction_log, session_id=session_id, user_id=user_id, error="Server configuration error: RunConfig missing.")
    
    async for event_obj in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_input_content,
        run_config=current_run_config
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
                            final_agent_utterance = part_data["text"]
                            full_interaction_log.append(f"Agent: {final_agent_utterance}")
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

    return ConverseResponse(
        final_utterance=final_agent_utterance or "No text response from agent.",
        full_log=full_interaction_log,
        session_id=session_id,
        user_id=user_id,
        error=error_message_from_event
    )

@app.post("/start_agent_interaction/", response_model=StartAgentResponse)
async def start_agent_interaction(fastapi_request: FastAPIRequest, request: StartAgentRequest = Body(...)):
    agent_name = request.agent_name
    user_id = API_DEFAULT_USER_ID
    print(f"Request to start session for agent='{agent_name}', user='{user_id}'")
    try:
        agent = get_agent_instance(agent_name)

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

        initial_utterance = None
        error_from_greeting: Optional[str] = None
        
        query_endpoint_path = fastapi_request.url_for('agent_converse_turn', agent_name_in_path=agent_name, session_id_in_path=current_session_id)
        
        if request.initial_greeting_query is not None:
            print(f"Running initial greeting for new session '{current_session_id}': '{request.initial_greeting_query}'")
            greeting_response = await _run_agent_turn_for_session(
                agent=agent, session_id=current_session_id, user_id=user_id,
                app_name=agent_name, query_text=request.initial_greeting_query
            )
            initial_utterance = greeting_response.final_utterance
            error_from_greeting = greeting_response.error
            if error_from_greeting:
                 print(f"Error during initial greeting for session {current_session_id}: {error_from_greeting}")

        return StartAgentResponse(
            agent_name=agent_name,
            session_id=current_session_id,
            user_id=user_id,
            query_endpoint=str(query_endpoint_path),
            initial_agent_utterance=initial_utterance,
            error=error_from_greeting
        )

    except (ValueError, ImportError, AttributeError, NotImplementedError) as e:
        print(f"Error starting session for agent '{agent_name}': {str(e)}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Could not start session: {str(e)}")
    except Exception as e:
        print(f"Unexpected error starting session: {str(e)}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/chat/{agent_name_in_path}/{session_id_in_path}/query", response_model=ConverseResponse, name="agent_converse_turn")
async def agent_converse_turn(
    agent_name_in_path: str = Path(..., description="The name of the agent to interact with"),
    session_id_in_path: str = Path(..., description="The active session ID for this conversation"),
    request: ConverseRequest = Body(...)
):
    query = request.query
    user_id = API_DEFAULT_USER_ID

    print(f"Converse turn: agent='{agent_name_in_path}', session='{session_id_in_path}', query='{query}', user='{user_id}'")
    
    try:
        loaded_session: Optional[Session] = None
        if hasattr(session_service, "get_session"):
            if asyncio.iscoroutinefunction(session_service.get_session):
                loaded_session = await session_service.get_session(
                    session_id=session_id_in_path, user_id=user_id, app_name=agent_name_in_path
                )
            else:
                loaded_session = session_service.get_session(
                    session_id=session_id_in_path, user_id=user_id, app_name=agent_name_in_path
                )
        
        if not loaded_session:
            raise ValueError(f"Session ID '{session_id_in_path}' not found for user '{user_id}' and agent '{agent_name_in_path}'. Please start a new session via /start_agent_interaction/.")

        agent = get_agent_instance(agent_name_in_path)

        return await _run_agent_turn_for_session(
            agent=agent,
            session_id=session_id_in_path,
            user_id=user_id,
            app_name=agent_name_in_path,
            query_text=query
        )

    except ValueError as e:
        print(f"ValueError processing query for session '{session_id_in_path}': {str(e)}")
        import traceback; traceback.print_exc()
        return ConverseResponse(final_utterance=None, full_log=[f"User: {query}", f"Error: {str(e)}"], session_id=session_id_in_path, user_id=user_id, error=str(e))
    except Exception as e:
        print(f"Unexpected error processing query: {str(e)}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Dynamic Endpoint ADK Agent Service...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
