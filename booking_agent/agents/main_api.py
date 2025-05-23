from fastapi import FastAPI, HTTPException, Body, Path, Header, Request as FastAPIRequest
from pydantic import BaseModel
from typing import Any, Optional, Dict, AsyncGenerator
import importlib
import os
from dotenv import load_dotenv
import asyncio
from contextlib import asynccontextmanager, AsyncExitStack # Added AsyncExitStack

from google.adk import Agent, Runner
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.sessions import Session
from google.adk.agents.run_config import RunConfig
from google.genai.types import Content

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, '.env')

load_dotenv()
ADK_SQL_DB_URL = os.getenv(
    "ADK_SQL_DB_URL",
    f"sqlite:///{os.path.join(BASE_DIR, 'adk_internal_sessions.db')}"
)
print(f"ADK Internal Session DB URL (SQLite): {ADK_SQL_DB_URL}")

adk_sql_session_service: Optional[DatabaseSessionService] = None

AGENT_MODULE_PATHS = {
    "movie_ticket_agent": "Movie_ticket_booking_agent.agent",
    "bus_booking_agent": "Bus_ticket_booking_agent.agent",
}
loaded_agents: Dict[str, Agent] = {}
application_level_exit_stack: Optional[AsyncExitStack] = None # For managing tool exit_stacks
API_DEFAULT_USER_ID = "user_1"

@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
    print("Application startup: Initializing ADK DatabaseSessionService and Agents...")
    global adk_sql_session_service, loaded_agents, application_level_exit_stack
    application_level_exit_stack = AsyncExitStack()

    try:
        print(f"Initializing ADK DatabaseSessionService with SQL URL: {ADK_SQL_DB_URL}")
        adk_sql_session_service = DatabaseSessionService(db_url=ADK_SQL_DB_URL)
        print(f"ADK DatabaseSessionService (for SQL) initialized.")
    except NameError:
        print("ERROR: DatabaseSessionService class not imported correctly. Check import path.")
        raise RuntimeError("Failed to initialize ADK DatabaseSessionService due to import error.")
    except Exception as e_adk_sess_init:
        print(f"ERROR: Could not instantiate ADK DatabaseSessionService: {e_adk_sess_init}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to initialize ADK DatabaseSessionService: {e_adk_sess_init}")

    print("Loading agents...")
    for agent_name, module_path in AGENT_MODULE_PATHS.items():
        try:
            print(f"Attempting to load agent: {agent_name} from module: {module_path}")
            agent_module = importlib.import_module(module_path)
            
            agent_coroutine = None
            if hasattr(agent_module, "root_agent"): # root_agent = create_root() pattern
                agent_coroutine = agent_module.root_agent
            elif hasattr(agent_module, "create_root") and asyncio.iscoroutinefunction(agent_module.create_root):
                agent_coroutine = agent_module.create_root() # Call to get coroutine
            
            if asyncio.iscoroutine(agent_coroutine):
                # Expects create_root() to return (Agent, exit_stack_for_tools)
                agent_instance, tool_exit_stack = await agent_coroutine
                loaded_agents[agent_name] = agent_instance
                if tool_exit_stack and application_level_exit_stack: # MCPToolset returns an exit_stack
                    await application_level_exit_stack.enter_async_context(tool_exit_stack)
                print(f"Agent '{agent_name}' loaded successfully and its toolset exit_stack (if any) entered.")
            elif isinstance(agent_coroutine, Agent): # If root_agent was already an instance
                 loaded_agents[agent_name] = agent_coroutine
                 print(f"Agent '{agent_name}' (pre-instantiated) loaded.")
            else:
                raise AttributeError(f"Module {module_path} for agent '{agent_name}' does not provide a usable 'root_agent' or 'create_root'.")

        except Exception as e_agent_load:
            print(f"ERROR: Failed to load agent '{agent_name}': {e_agent_load}")
            import traceback
            traceback.print_exc()
            # Decide if app should fail hard if an agent fails to load. For now, it continues.

    print("Application startup complete.")
    yield
    print("Application shutdown: Cleaning up...")
    if application_level_exit_stack:
        print("Exiting application-level agent toolset exit_stacks...")
        await application_level_exit_stack.aclose()
        print("Agent toolset exit_stacks closed.")
    
    if adk_sql_session_service and hasattr(adk_sql_session_service, "dispose_engine"):
        print("Disposing ADK DatabaseSessionService engine...")
        # If dispose_engine is async, await it. Assuming sync or handled by ADK.
    print("Application shutdown complete.")

app = FastAPI(
    title="ADK Agent Service (SQLite ADK Sessions)",
    lifespan=lifespan
)

def get_agent_instance(agent_name: str) -> Agent:
    if agent_name in loaded_agents:
        return loaded_agents[agent_name]
    # This path should ideally not be hit if agents are pre-loaded in lifespan
    raise ValueError(
        f"Agent '{agent_name}' was not found in pre-loaded agents. "
        "Ensure it is configured in AGENT_MODULE_PATHS and loads correctly during startup."
    )

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
    final_agent_utterance: Optional[str] = None
    session_id: str
    user_id: str
    error_message: Optional[str] = None

async def _run_single_agent_turn(
    agent_to_run: Agent,
    session_id: str,
    user_id: str,
    app_name_for_run: str,
    query_text: str,
) -> Dict[str, Any]:
    global adk_sql_session_service
    if not adk_sql_session_service:
        print("ERROR: ADK DatabaseSessionService (SQL) not initialized.")
        return {"final_utterance": "Server Error: ADK Session service unavailable.",
                 "session_id": session_id, "user_id": user_id,
                 "error": "ADK Session service unavailable."}

    runner = Runner(
        app_name=app_name_for_run,
        agent=agent_to_run, # This is now guaranteed to be an Agent instance
        session_service=adk_sql_session_service
    )

    new_turn_message = Content(parts=[{"text": query_text}], role="user")
    print(f"Runner for '{app_name_for_run}' created. Session: '{session_id}'. Sending message.")

    _final_agent_utterance: Optional[str] = None
    error_message_from_event: Optional[str] = None

    try:
        current_run_config = RunConfig()
    except NameError:
        error_msg = "Server configuration error: RunConfig missing."
        return {"final_utterance": None,
                "session_id": session_id, "user_id": user_id, "error": error_msg}

    try:
        async for event_obj in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_turn_message,
            run_config=current_run_config
        ):
            try:
                event_data_for_log = event_obj.model_dump(exclude_none=True)
            except AttributeError:
                event_data_for_log = event_obj.dict(exclude_none=True)

            print(f"Event Data ({app_name_for_run}): {event_data_for_log}")

            content = event_data_for_log.get("content")
            if content and isinstance(content, dict):
                parts = content.get("parts")
                role = content.get("role")
                if role == "model" and parts and isinstance(parts, list):
                    for part_data in parts:
                        if isinstance(part_data, dict):
                            if "text" in part_data and part_data["text"]:
                                _final_agent_utterance = part_data["text"]

            if hasattr(event_obj, "error_details") and event_obj.error_details:
                error_message_from_event = str(event_obj.error_details)
            elif event_data_for_log.get("error_message") or event_data_for_log.get("error"):
                error_message_from_event = str(event_data_for_log.get("error_message") or event_data_for_log.get("error"))

            if error_message_from_event:
                print(f"Error event received from ADK: {error_message_from_event}")
                break
    except Exception as e_run:
        print(f"Exception during runner.run_async for {app_name_for_run}: {e_run}")
        import traceback; traceback.print_exc()
        error_message_from_event = f"Agent execution failed: {str(e_run)}"

    return {
        "final_utterance": _final_agent_utterance,
        "session_id": session_id,
        "user_id": user_id,
        "error": error_message_from_event
    }

BUS_BOOKING_AGENT_CARD = {
    "agent_name": "bus_booking_agent",
    "version": "1.0.0",
    "description": "Handles bus ticket booking queries.",
    "capabilities_summary": [
        "finding bus routes", "checking bus seat availability", "booking bus tickets"
    ],
    "endpoints": {
        "execute_task": {
            "url": f"{os.getenv('MAIN_API_BASE_URL', 'http://localhost:8001/bus_booking_agent')}/query",
            "method": "POST",
            "request_schema_summary": "{'query': 'string'} (expects X-Session-Id in header for this specific endpoint)",
            "response_schema_summary": "{'final_agent_utterance': 'string', ...}"
        }
    }
}
MOVIE_TICKET_AGENT_CARD = {
    "agent_name": "movie_ticket_agent",
    "version": "1.0.0",
    "description": "Handles all movie ticket booking related queries.",
    "capabilities_summary": [
        "finding movie showtimes", "selecting seats", "confirming movie bookings"
    ],
    "endpoints": {
        "execute_task": {
            "url": f"{os.getenv('MAIN_API_BASE_URL', 'http://localhost:8001/movie_ticket_agent')}/query",
            "method": "POST",
            "request_schema_summary": "{'query': 'string'} (expects X-Session-Id in header for this specific endpoint)",
            "response_schema_summary": "{'final_agent_utterance': 'string', ...}"
        }
    }
}

@app.post("/start_agent_interaction/", response_model=StartAgentResponse)
async def start_agent_interaction(fastapi_request: FastAPIRequest, request: StartAgentRequest = Body(...)):
    global adk_sql_session_service
    agent_name = request.agent_name
    user_id = API_DEFAULT_USER_ID
    print(f"Request to start session for agent='{agent_name}', user='{user_id}'")

    if not adk_sql_session_service:
        raise HTTPException(status_code=503, detail="ADK Session service not available.")
    try:
        # Agent is now retrieved from pre-loaded cache, guaranteed to be an Agent instance
        _agent = get_agent_instance(agent_name)

        print(f"Creating new ADK session (SQL-backed): user_id='{user_id}', app_name='{agent_name}'")
        created_session_obj: Optional[Session] = None
        if hasattr(adk_sql_session_service, "create_session"):
            if asyncio.iscoroutinefunction(adk_sql_session_service.create_session):
                created_session_obj = await adk_sql_session_service.create_session(
                    app_name=agent_name, user_id=user_id
                )
            else:
                created_session_obj = adk_sql_session_service.create_session(
                    app_name=agent_name, user_id=user_id
                )
            if not created_session_obj or not hasattr(created_session_obj, 'id') or not created_session_obj.id:
                 raise RuntimeError(f"ADK DatabaseSessionService 'create_session' failed.")
            current_session_id = created_session_obj.id
            print(f"New ADK session '{current_session_id}' (SQL-backed) created for app '{agent_name}'.")
        else:
            raise NotImplementedError("ADK DatabaseSessionService instance lacks 'create_session' method.")

        query_endpoint_url_path = fastapi_request.url_for(
            'agent_query_turn_handler', agent_name_in_path=agent_name
        )

        return StartAgentResponse(
            agent_name=agent_name, session_id=current_session_id,
            user_id=user_id, query_endpoint=str(query_endpoint_url_path)
        )
    except (ValueError, ImportError, AttributeError, NotImplementedError, RuntimeError) as e:
        print(f"Error starting session for agent '{agent_name}': {str(e)}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Could not start session: {str(e)}")
    except Exception as e:
        print(f"Unexpected error starting session: {str(e)}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/{agent_name_in_path}/query", response_model=ConverseResponse, name="agent_query_turn_handler")
async def agent_query_turn_handler(
    agent_name_in_path: str = Path(..., description="The agent to interact with"),
    x_session_id: str = Header(..., alias="X-Session-Id", description="The active session ID"),
    request: ConverseRequest = Body(...)
):
    global adk_sql_session_service
    query = request.query
    agent_name = agent_name_in_path
    session_id = x_session_id
    user_id = API_DEFAULT_USER_ID

    print(f"Converse turn: agent='{agent_name}', session='{session_id}', query='{query}', user='{user_id}'")
    if not adk_sql_session_service:
        raise HTTPException(status_code=503, detail="ADK Session service not available.")

    try:
        loaded_adk_session: Optional[Session] = None
        if hasattr(adk_sql_session_service, "get_session"):
            if asyncio.iscoroutinefunction(adk_sql_session_service.get_session):
                loaded_adk_session = await adk_sql_session_service.get_session(
                    session_id=session_id, user_id=user_id, app_name=agent_name
                )
            else:
                loaded_adk_session = adk_sql_session_service.get_session(
                    session_id=session_id, user_id=user_id, app_name=agent_name
                )
        if not loaded_adk_session:
            raise ValueError(f"ADK Session ID '{session_id}' not found for user '{user_id}' and agent '{agent_name}'.")

        # Agent is now retrieved from pre-loaded cache
        agent_instance = get_agent_instance(agent_name)
        print(f"Agent '{agent_name}' retrieved for ADK session '{session_id}'.")

        turn_result_dict = await _run_single_agent_turn(
            agent_to_run=agent_instance,
            session_id=session_id,
            user_id=user_id,
            app_name_for_run=agent_name,
            query_text=query
        )

        return ConverseResponse(
            final_agent_utterance=turn_result_dict.get("final_utterance"),
            session_id=turn_result_dict["session_id"],
            user_id=turn_result_dict["user_id"],
            error_message=turn_result_dict.get("error")
        )
    except ValueError as e:
        print(f"ValueError processing query for session '{session_id}': {str(e)}")
        import traceback; traceback.print_exc()
        return ConverseResponse(session_id=session_id, user_id=user_id, error_message=str(e))
    except Exception as e:
        print(f"Unexpected error processing query: {str(e)}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/movie_booking_agent/.well-known/agent-card")
async def get_movie_ticket_agent_card():
    return MOVIE_TICKET_AGENT_CARD

@app.get("/bus_booking_agent/.well-known/agent-card")
async def get_bus_agent_card():
    return BUS_BOOKING_AGENT_CARD

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)