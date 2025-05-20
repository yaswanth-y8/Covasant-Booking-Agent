# -*- coding: utf-8 -*-
# client_agent_service_main.py

import os
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv # Still load .env here in case agent.py missed it or for service-specific vars
from typing import Optional, Dict, Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Body, Path, Header, Request as FastAPIRequest
from pydantic import BaseModel

# ADK imports
from google.adk.agents import Agent # For type checking
from google.adk.runners import Runner # For type checking
from google.adk.agents.run_config import RunConfig
from google.genai.types import Content

# --- Load environment variables specific to this ClientAgent service ---
# This ensures .env is loaded before agent.py might try to access os.getenv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, '.env')
if os.path.exists(DOTENV_PATH):
    load_dotenv(dotenv_path=DOTENV_PATH, override=True)
    print(f"[ClientAgentService - main.py] .env loaded from {DOTENV_PATH}.")
else:
    print(f"[ClientAgentService - main.py] Warning: .env file not found at {DOTENV_PATH}.")

# --- Import the PRE-CONFIGURED ClientAgent and its Runner ---
try:
    from agent import (
        root_agent as client_agent_instance, # The ADK Agent instance
        runner as client_agent_runner_instance, # The ADK Runner configured with DatabaseSessionService
        USER_ID as CLIENT_AGENT_DEFAULT_USER_ID, # Default user_id for this agent's operations
        APP_NAME as CLIENT_AGENT_DEFAULT_APP_NAME # Default app_name for this agent
    )
    if not isinstance(client_agent_instance, Agent):
        raise TypeError("Imported 'root_agent' is not an ADK Agent instance.")
    if not isinstance(client_agent_runner_instance, Runner):
        raise TypeError("Imported 'runner' is not an ADK Runner instance.")
    print("[ClientAgentService - main.py] Pre-configured ClientAgent and its Runner imported successfully.")
except ImportError as e:
    print(f"[ClientAgentService - main.py] CRITICAL ERROR: Could not import ClientAgent components from agent.py: {e}")
    raise
except TypeError as e:
    print(f"[ClientAgentService - main.py] CRITICAL ERROR: Imported components from agent.py have incorrect types: {e}")
    raise


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
    print("[ClientAgentService - main.py] Application startup...")
    # The DatabaseSessionService and Runner are now initialized in agent.py
    # Ensure agent.py's session_service (if it's DatabaseSessionService) is healthy
    if hasattr(client_agent_runner_instance.session_service, "db_url"):
         print(f"[ClientAgentService - main.py] ClientAgent Runner is using session service connected to: {client_agent_runner_instance.session_service.db_url}")
    else:
        print("[ClientAgentService - main.py] ClientAgent Runner is using InMemorySessionService (or unknown type).")
    print("[ClientAgentService - main.py] Application startup complete.")
    yield
    print("[ClientAgentService - main.py] Application shutdown...")
    print("[ClientAgentService - main.py] Application shutdown complete.")

app = FastAPI(
    title="ClientAgent HTTP Service (using DB Sessions from agent.py)",
    lifespan=lifespan
)

# --- Pydantic Models (remain the same) ---
class ClientStartSessionResponse(BaseModel):
    client_agent_name: str
    session_id: str
    user_id: str # The user_id for the session with THIS ClientAgent service
    query_endpoint: str

class ClientConverseRequest(BaseModel):
    query: str

class ClientConverseResponse(BaseModel):
    final_client_agent_utterance: Optional[str] = None
    session_id: str
    user_id: str
    error_message: Optional[str] = None

# --- Helper to run ClientAgent for a turn (uses imported runner) ---
async def _run_imported_client_agent_turn(
    session_id: str,
    user_id: str, # User interacting with this FastAPI service for this session
    query_text: str,
) -> Dict[str, Any]:
    # Use the globally imported and configured runner from agent.py
    global client_agent_runner_instance, CLIENT_AGENT_DEFAULT_APP_NAME

    # The app_name for the run should be the one the runner was configured with
    app_name_for_run = client_agent_runner_instance.app_name

    new_turn_message = Content(parts=[{"text": query_text}], role="user")
    print(f"[ClientAgentService] Using pre-configured Runner for '{app_name_for_run}'. Session: '{session_id}', User: '{user_id}'. Sending query.")

    _final_agent_utterance: Optional[str] = None
    error_message_from_event: Optional[str] = None

    try:
        current_run_config = RunConfig()
    except NameError:
        return {"final_utterance": None, "session_id": session_id, "user_id": user_id,
                "error": "Server configuration error: RunConfig missing."}
    try:
        async for event_obj in client_agent_runner_instance.run_async(
            user_id=user_id, # This user_id is for the session with ClientAgent itself
            session_id=session_id,
            new_message=new_turn_message,
            run_config=current_run_config
        ):
            try: event_data = event_obj.model_dump(exclude_none=True)
            except AttributeError: event_data = event_obj.dict(exclude_none=True)
            # print(f"[ClientAgentService Event Log]: {event_data}")
            if event_obj.is_final_response():
                if event_obj.content and event_obj.content.parts:
                    text_parts = [part.text for part in event_obj.content.parts if hasattr(part, 'text')]
                    _final_agent_utterance = " ".join(text_parts).strip()
                if hasattr(event_obj, "error_details") and event_obj.error_details:
                    error_message_from_event = str(event_obj.error_details)
                elif hasattr(event_obj, "error_message") and event_obj.error_message:
                    error_message_from_event = str(event_obj.error_message)
                if error_message_from_event: print(f"[ClientAgentService] Error event: {error_message_from_event}")
                break
    except Exception as e_run:
        print(f"[ClientAgentService] Exception during ClientAgent run_async: {e_run}")
        import traceback; traceback.print_exc()
        error_message_from_event = f"ClientAgent execution failed: {str(e_run)}"

    return {"final_utterance": _final_agent_utterance, "session_id": session_id, "user_id": user_id, "error": error_message_from_event}


# For sessions with this ClientAgent service
API_END_USER_DEFAULT_ID = "web_ui_user_001" # Represents the user talking to this FastAPI app

@app.post("/client/start_session", response_model=ClientStartSessionResponse)
async def start_client_agent_session(fastapi_request: FastAPIRequest):
    global client_agent_runner_instance, client_agent_instance # Use imported instances

    # The user_id for *this* ClientAgent's session in its own DB
    user_id_for_client_session = API_END_USER_DEFAULT_ID
    # The app_name for *this* ClientAgent's session (from its runner config)
    app_name_for_client_session = client_agent_runner_instance.app_name

    session_service_to_use = client_agent_runner_instance.session_service
    if not session_service_to_use: # Should not happen if runner is configured
        raise HTTPException(status_code=503, detail="ClientAgent Session service not configured in runner.")

    try:
        print(f"[ClientAgentService] Creating new session via runner's service for app='{app_name_for_client_session}', user='{user_id_for_client_session}'")
        
        # Use the session_service from the imported runner
        # Call create_session synchronously as established before for DatabaseSessionService
        created_session_obj = session_service_to_use.create_session(
            app_name=app_name_for_client_session,
            user_id=user_id_for_client_session
        )
        
        if not created_session_obj or not hasattr(created_session_obj, 'id') or not created_session_obj.id:
            raise RuntimeError("ClientAgent's configured SessionService 'create_session' failed.")
        
        current_session_id = created_session_obj.id
        print(f"[ClientAgentService] New session '{current_session_id}' created for ClientAgent by its runner's session service.")

        query_endpoint_url_path = fastapi_request.url_for('client_agent_query_handler')

        return ClientStartSessionResponse(
            client_agent_name=client_agent_instance.name,
            session_id=current_session_id,
            user_id=user_id_for_client_session,
            query_endpoint=str(query_endpoint_url_path)
        )
    except Exception as e:
        print(f"[ClientAgentService] Error starting ClientAgent session: {str(e)}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Could not start ClientAgent session: {str(e)}")

@app.post("/client/query", response_model=ClientConverseResponse, name="client_agent_query_handler")
async def client_agent_query_handler(
    x_session_id: str = Header(..., alias="X-Session-Id", description="Active session ID for ClientAgent"),
    request: ClientConverseRequest = Body(...)
):
    global client_agent_runner_instance

    query = request.query
    session_id_from_header = x_session_id
    user_id_for_client_interaction = API_END_USER_DEFAULT_ID # The user of this API

    session_service_to_use = client_agent_runner_instance.session_service
    app_name_for_client_session = client_agent_runner_instance.app_name

    if not session_service_to_use:
        raise HTTPException(status_code=503, detail="ClientAgent Session service not configured in runner.")
    
    print(f"[ClientAgentService] Converse: session='{session_id_from_header}', query='{query}', user='{user_id_for_client_interaction}' for app '{app_name_for_client_session}'")
    try:
        # Validate session exists using the runner's session service
        # Call get_session synchronously
        loaded_adk_session = session_service_to_use.get_session(
            session_id=session_id_from_header,
            user_id=user_id_for_client_interaction,
            app_name=app_name_for_client_session
        )
        if not loaded_adk_session:
             raise ValueError(f"ClientAgent Session ID '{session_id_from_header}' not found for user '{user_id_for_client_interaction}'.")

        print(f"[ClientAgentService] Valid session '{session_id_from_header}' retrieved for ClientAgent.")

        turn_result_dict = await _run_imported_client_agent_turn(
            session_id=session_id_from_header,
            user_id=user_id_for_client_interaction,
            query_text=query
        )

        return ClientConverseResponse(
            final_client_agent_utterance=turn_result_dict.get("final_utterance"),
            session_id=turn_result_dict["session_id"],
            user_id=turn_result_dict["user_id"],
            error_message=turn_result_dict.get("error")
        )
    except ValueError as e:
        print(f"[ClientAgentService] ValueError processing query for ClientAgent session '{session_id_from_header}': {str(e)}")
        return ClientConverseResponse(session_id=session_id_from_header, user_id=user_id_for_client_interaction, error_message=str(e))
    except Exception as e:
        print(f"[ClientAgentService] Unexpected error processing query for ClientAgent: {str(e)}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing ClientAgent query: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("[ClientAgentService] Starting ClientAgent HTTP Service...")
    print(f"[ClientAgentService] This service will expose '{client_agent_instance.name}' (from agent.py).")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("CLIENT_AGENT_PORT", "8003")))