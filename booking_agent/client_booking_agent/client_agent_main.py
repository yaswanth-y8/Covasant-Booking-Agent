# -*- coding: utf-8 -*-
# client_agent_service_main.py

import os
import asyncio # Still needed for FastAPI and runner.run_async
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from typing import Optional, Dict, Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Body, Path, Header, Request as FastAPIRequest
from pydantic import BaseModel

from google.adk.agents import Agent
from google.adk.sessions import Session
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.runners import Runner
from google.adk.agents.run_config import RunConfig
from google.genai.types import Content

# ... (dotenv loading, CLIENT_AGENT_DB_URL - remain the same) ...
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, '.env')
if os.path.exists(DOTENV_PATH):
    load_dotenv(dotenv_path=DOTENV_PATH, override=True)
    print(f"[ClientAgentService] .env loaded from {DOTENV_PATH}.")
else:
    print(f"[ClientAgentService] Warning: .env file not found at {DOTENV_PATH}.")

CLIENT_AGENT_DB_URL = os.getenv(
    "CLIENT_AGENT_ADK_SQL_DB_URL",
    f"sqlite:///{os.path.join(BASE_DIR, 'client_agent_sessions.db')}"
)
print(f"[ClientAgentService] ClientAgent Session DB URL: {CLIENT_AGENT_DB_URL}")

client_agent_sql_session_service: Optional[DatabaseSessionService] = None
client_agent_adk_runner: Optional[Runner] = None

try:
    from client_agent import (
        client_agent as imported_client_agent_object,
        APP_NAME as CLIENT_AGENT_APP_NAME_CONFIG,
        USER_ID as CLIENT_AGENT_USER_ID_CONFIG
    )
    if not isinstance(imported_client_agent_object, Agent):
        raise TypeError("Imported 'client_agent' is not an ADK Agent instance.")
    print("[ClientAgentService] ClientAgent definition imported successfully.")
except ImportError as e:
    print(f"[ClientAgentService] CRITICAL ERROR: Could not import ClientAgent definition: {e}")
    raise
except TypeError as e:
    print(f"[ClientAgentService] CRITICAL ERROR: Imported ClientAgent component has incorrect type: {e}")
    raise


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
    print("[ClientAgentService] App startup: Initializing DatabaseSessionService for ClientAgent...")
    global client_agent_sql_session_service, client_agent_adk_runner
    global imported_client_agent_object, CLIENT_AGENT_APP_NAME_CONFIG

    try:
        client_agent_sql_session_service = DatabaseSessionService(db_url=CLIENT_AGENT_DB_URL)
        print(f"[ClientAgentService] DatabaseSessionService for ClientAgent initialized with URL: {CLIENT_AGENT_DB_URL}")

        client_agent_adk_runner = Runner(
            agent=imported_client_agent_object,
            app_name=CLIENT_AGENT_APP_NAME_CONFIG,
            session_service=client_agent_sql_session_service # Pass the synchronous service instance
        )
        print(f"[ClientAgentService] ADK Runner for '{imported_client_agent_object.name}' initialized with DatabaseSessionService.")

    except Exception as e_init:
        print(f"[ClientAgentService] ERROR during startup: {e_init}")
        import traceback; traceback.print_exc()
        raise RuntimeError(f"Failed to initialize ClientAgent service: {e_init}")

    print("[ClientAgentService] Application startup complete.")
    yield
    print("[ClientAgentService] Application shutdown...")
    print("[ClientAgentService] Application shutdown complete.")

app = FastAPI(
    title="ClientAgent HTTP Service (DB Sessions)",
    lifespan=lifespan
)

# Pydantic Models remain the same
class ClientStartSessionResponse(BaseModel):
    client_agent_name: str
    session_id: str
    user_id: str
    query_endpoint: str

class ClientConverseRequest(BaseModel):
    query: str

class ClientConverseResponse(BaseModel):
    final_client_agent_utterance: Optional[str] = None
    session_id: str
    user_id: str
    error_message: Optional[str] = None


# Helper _run_client_agent_turn_with_db_session remains the same (it uses runner.run_async which IS async)
async def _run_client_agent_turn_with_db_session(
    session_id: str,
    user_id: str,
    query_text: str,
) -> Dict[str, Any]:
    global client_agent_adk_runner, CLIENT_AGENT_APP_NAME_CONFIG

    if not client_agent_adk_runner:
        return {"final_utterance": "Server Error: ClientAgent Runner not initialized.",
                "session_id": session_id, "user_id": user_id,
                "error": "ClientAgent Runner not initialized."}

    new_turn_message = Content(parts=[{"text": query_text}], role="user")
    print(f"[ClientAgentService] DB-Session Runner for '{CLIENT_AGENT_APP_NAME_CONFIG}'. Session: '{session_id}', User: '{user_id}'. Sending query.")
    _final_agent_utterance: Optional[str] = None
    error_message_from_event: Optional[str] = None
    try:
        current_run_config = RunConfig()
    except NameError:
        return {"final_utterance": None, "session_id": session_id, "user_id": user_id,
                "error": "Server configuration error: RunConfig missing."}
    try:
        async for event_obj in client_agent_adk_runner.run_async( # This call is async
            user_id=user_id,
            session_id=session_id,
            new_message=new_turn_message,
            run_config=current_run_config
        ):
            try: event_data = event_obj.model_dump(exclude_none=True)
            except AttributeError: event_data = event_obj.dict(exclude_none=True)
            # print(f"[ClientAgent DB-Session Event]: Author: {event_obj.author if hasattr(event_obj, 'author') else 'N/A'}, Data: {event_data}")
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
        print(f"[ClientAgentService] Exception during ClientAgent run_async (DB session): {e_run}")
        import traceback; traceback.print_exc()
        error_message_from_event = f"ClientAgent execution failed: {str(e_run)}"
    return {"final_utterance": _final_agent_utterance, "session_id": session_id, "user_id": user_id, "error": error_message_from_event}


API_CLIENT_DEFAULT_USER_ID = "end_user_001"

@app.post("/client/start_session", response_model=ClientStartSessionResponse)
async def start_client_agent_session_db(fastapi_request: FastAPIRequest):
    global client_agent_sql_session_service, imported_client_agent_object, CLIENT_AGENT_APP_NAME_CONFIG
    
    user_id_for_session = API_CLIENT_DEFAULT_USER_ID
    app_name_for_session = CLIENT_AGENT_APP_NAME_CONFIG

    if not client_agent_sql_session_service:
        raise HTTPException(status_code=503, detail="ClientAgent Session service not available.")
    try:
        print(f"[ClientAgentService] Creating new DB session for ClientAgent: user_id='{user_id_for_session}', app_name='{app_name_for_session}'")
        
        # MODIFICATION: Call create_session synchronously
        created_session_obj: Optional[Session] = client_agent_sql_session_service.create_session(
            app_name=app_name_for_session,
            user_id=user_id_for_session
        )
        # No 'await' here ^^^
        
        if not created_session_obj or not hasattr(created_session_obj, 'id') or not created_session_obj.id:
            raise RuntimeError("ClientAgent's DatabaseSessionService 'create_session' failed to return a valid session object or ID.")
        
        current_session_id = created_session_obj.id
        print(f"[ClientAgentService] New DB session '{current_session_id}' created for ClientAgent.")

        query_endpoint_url_path = fastapi_request.url_for('client_agent_query_handler_db')

        return ClientStartSessionResponse(
            client_agent_name=imported_client_agent_object.name,
            session_id=current_session_id,
            user_id=user_id_for_session,
            query_endpoint=str(query_endpoint_url_path)
        )
    except Exception as e:
        print(f"[ClientAgentService] Error starting ClientAgent DB session: {str(e)}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Could not start ClientAgent DB session: {str(e)}")

@app.post("/client/query", response_model=ClientConverseResponse, name="client_agent_query_handler_db")
async def client_agent_query_handler_db(
    x_session_id: str = Header(..., alias="X-Session-Id", description="Active session ID for ClientAgent"),
    request: ClientConverseRequest = Body(...)
):
    global client_agent_sql_session_service, CLIENT_AGENT_APP_NAME_CONFIG, CLIENT_AGENT_USER_ID_CONFIG

    query = request.query
    session_id = x_session_id
    user_id_for_this_interaction = API_CLIENT_DEFAULT_USER_ID # The user talking to the API

    if not client_agent_sql_session_service:
        raise HTTPException(status_code=503, detail="ClientAgent Session service not available.")
    
    print(f"[ClientAgentService] DB Converse: session='{session_id}', query='{query}', user='{user_id_for_this_interaction}' for app '{CLIENT_AGENT_APP_NAME_CONFIG}'")
    try:
        # MODIFICATION: Call get_session synchronously
        # Use the user_id that would have been used to create THIS session for THIS ClientAgent app_name
        loaded_adk_session: Optional[Session] = client_agent_sql_session_service.get_session(
            session_id=session_id, 
            user_id=user_id_for_this_interaction, # Use the API user for retrieving their session
            app_name=CLIENT_AGENT_APP_NAME_CONFIG
        )
        # No 'await' here ^^^

        if not loaded_adk_session:
            # It's possible the session was created with CLIENT_AGENT_USER_ID_CONFIG if API_CLIENT_DEFAULT_USER_ID was different
            # Check with CLIENT_AGENT_USER_ID_CONFIG if it wasn't found with API_CLIENT_DEFAULT_USER_ID
            print(f"[ClientAgentService] Session '{session_id}' not found for user '{user_id_for_this_interaction}'. Trying with default agent user ID '{CLIENT_AGENT_USER_ID_CONFIG}'.")
            loaded_adk_session = client_agent_sql_session_service.get_session(
                session_id=session_id, 
                user_id=CLIENT_AGENT_USER_ID_CONFIG, # Try with the fixed User ID used by this ClientAgent's runner if different
                app_name=CLIENT_AGENT_APP_NAME_CONFIG
            )
            if not loaded_adk_session:
                 raise ValueError(f"ClientAgent Session ID '{session_id}' not found for any known user context.")
            # If found, use the user_id that the session was actually found with for the run
            user_id_for_run = CLIENT_AGENT_USER_ID_CONFIG
        else:
            user_id_for_run = user_id_for_this_interaction


        print(f"[ClientAgentService] Valid DB session '{session_id}' retrieved for ClientAgent (User: {user_id_for_run}).")

        turn_result_dict = await _run_client_agent_turn_with_db_session(
            session_id=session_id,
            user_id=user_id_for_run, 
            query_text=query
        )

        return ClientConverseResponse(
            final_client_agent_utterance=turn_result_dict.get("final_utterance"),
            session_id=turn_result_dict["session_id"],
            user_id=turn_result_dict["user_id"], # Reflects the user_id used for the run
            error_message=turn_result_dict.get("error")
        )
    # ... (exception handling remains the same) ...
    except ValueError as e:
        print(f"[ClientAgentService] ValueError processing query for ClientAgent session '{session_id}': {str(e)}")
        return ClientConverseResponse(session_id=session_id, user_id=user_id_for_this_interaction, error_message=str(e))
    except Exception as e:
        print(f"[ClientAgentService] Unexpected error processing query for ClientAgent: {str(e)}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing ClientAgent query: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("[ClientAgentService] Starting ClientAgent HTTP Service with DatabaseSessionService...")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("CLIENT_AGENT_PORT", "8003")))