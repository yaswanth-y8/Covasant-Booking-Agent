import os
import json
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import httpx
from google.adk.agents import Agent
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.runners import Runner

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, '.env')
if os.path.exists(DOTENV_PATH):
    load_dotenv(dotenv_path=DOTENV_PATH, override=True)
    print(f"[ClientAgent Definition - agent.py] .env loaded from {DOTENV_PATH}.")
else:
    print(f"[ClientAgent Definition - agent.py] Warning: .env file not found at {DOTENV_PATH}.")

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

import google.generativeai as genai
genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-1.5-flash-latest"
print(f"[ClientAgent Definition - agent.py] Using Google Model: {MODEL_NAME}")

CLIENT_AGENT_DB_URL_FROM_ENV = os.getenv(
    "CLIENT_AGENT_ADK_SQL_DB_URL",
    f"sqlite:///{os.path.join(BASE_DIR, 'client_agent_sessions.db')}"
)
print(f"[ClientAgent Definition - agent.py] ClientAgent's own Session DB URL: {CLIENT_AGENT_DB_URL_FROM_ENV}")

SPECIALIST_AGENT_DISCOVERY_BASE_URLS = []
movie_agent_discovery_base = os.getenv("MOVIE_BOOKING_AGENT_DISCOVERY_BASE_URL")
if movie_agent_discovery_base:
    SPECIALIST_AGENT_DISCOVERY_BASE_URLS.append(movie_agent_discovery_base)

bus_agent_discovery_base = os.getenv("BUS_BOOKING_AGENT_DISCOVERY_BASE_URL")
if bus_agent_discovery_base:
    SPECIALIST_AGENT_DISCOVERY_BASE_URLS.append(bus_agent_discovery_base)

if not SPECIALIST_AGENT_DISCOVERY_BASE_URLS:
    print("[ClientAgent Definition - agent.py] WARNING: No specialist agent discovery base URLs configured.")

MAIN_API_SERVER_BASE_URL_FOR_SESSIONS = os.getenv("MAIN_API_SERVER_BASE_URL")
if not MAIN_API_SERVER_BASE_URL_FOR_SESSIONS:
    print("[ClientAgent Definition - agent.py] WARNING: MAIN_API_SERVER_BASE_URL not set in .env. Session start for specialists might fail.")
print(f"[ClientAgent Definition - agent.py] MAIN_API_SERVER_BASE_URL (for specialist sessions): {MAIN_API_SERVER_BASE_URL_FOR_SESSIONS}")


instruction_template = (
    "You are a master routing agent..."
    "`start_session_with_specialist(specialist_agent_name: str, specialist_base_api_url: str)` - to get a session_id FOR THE SPECIALIST. For `specialist_base_api_url`, you MUST use the value '{param_specialist_api_base_url}'. "
    "Known specialist agent DISCOVERY base URLs (for fetch_agent_card tool) are: {param_discovery_urls}.\n"

    "    a.  Your NEXT tool call MUST be to `start_session_with_specialist`. Provide its `agent_name` (e.g., 'movie_booking_agent') and for the `specialist_base_api_url` argument, you MUST pass the string '{param_specialist_api_base_url}'. You will get a `specialist_session_id` string as the tool response.\n"

)

_client_agent_instruction = instruction_template.format(
    param_specialist_api_base_url=MAIN_API_SERVER_BASE_URL_FOR_SESSIONS,
    param_discovery_urls=', '.join(SPECIALIST_AGENT_DISCOVERY_BASE_URLS) if SPECIALIST_AGENT_DISCOVERY_BASE_URLS else 'None configured'
)

AGENT_CARDS_CACHE: Dict[str, Dict] = {}
SPECIALIST_SESSIONS_CACHE: Dict[str, str] = {}

async def fetch_agent_card(agent_discovery_base_url: str) -> Optional[Dict]:
    if agent_discovery_base_url in AGENT_CARDS_CACHE:
        print(f"[ClientAgentDefinition Tool] Returning cached card for {agent_discovery_base_url}")
        return AGENT_CARDS_CACHE[agent_discovery_base_url]
    try:
        card_url = f"{agent_discovery_base_url}/.well-known/agent-card"
        async with httpx.AsyncClient(timeout=5.0) as client:
            print(f"[ClientAgentDefinition Tool] Fetching card from {card_url}...")
            response = await client.get(card_url)
            response.raise_for_status()
            card = response.json()
            AGENT_CARDS_CACHE[agent_discovery_base_url] = card
            print(f"[ClientAgentDefinition Tool] Fetched card for {card.get('agent_name')} from {agent_discovery_base_url}")
            return card
    except Exception as e:
        print(f"[ClientAgentDefinition Tool] Error fetching card from {agent_discovery_base_url}: {e}")
        return {"error": f"Could not fetch card from {agent_discovery_base_url}: {str(e)}"}

async def start_session_with_specialist(
    specialist_agent_name: str,
    specialist_base_api_url: str
) -> str:
    if not specialist_base_api_url:
        error_msg = "Error: `specialist_base_api_url` was not provided to the `start_session_with_specialist` tool."
        print(f"[ClientAgentDefinition Tool] {error_msg}")
        return error_msg

    if specialist_agent_name in SPECIALIST_SESSIONS_CACHE:
        cached_session_id = SPECIALIST_SESSIONS_CACHE[specialist_agent_name]
        if not cached_session_id.startswith("Error:"):
            print(f"[ClientAgentDefinition Tool] Using cached session for {specialist_agent_name}: {cached_session_id}")
            return cached_session_id
        else:
            print(f"[ClientAgentDefinition Tool] Cached session for {specialist_agent_name} was an error, attempting to get new one.")
    try:
        start_session_url = f"{specialist_base_api_url}/start_agent_interaction/"
        payload = {"agent_name": specialist_agent_name}
        print(f"[ClientAgentDefinition Tool] Starting session with {specialist_agent_name} at {start_session_url} using base URL '{specialist_base_api_url}'...")
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(start_session_url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            print(f"[ClientAgentDefinition Tool] Session started with {specialist_agent_name}: {response_data}")
            session_id = response_data.get("session_id")
            if session_id:
                SPECIALIST_SESSIONS_CACHE[specialist_agent_name] = session_id
                return session_id
            else:
                error_msg = f"Error: Could not get session_id from {specialist_agent_name}. Response: {response_data}"
                print(error_msg)
                SPECIALIST_SESSIONS_CACHE[specialist_agent_name] = error_msg
                return error_msg
    except Exception as e:
        error_msg = f"Error: Failed to start session with {specialist_agent_name}: {str(e)}"
        print(f"[ClientAgentDefinition Tool] {error_msg}")
        SPECIALIST_SESSIONS_CACHE[specialist_agent_name] = error_msg
        return error_msg

async def delegate_task_with_session(
    target_agent_name: str,
    target_agent_execute_url: str,
    specialist_session_id: str,
    payload: Dict[str, Any]
) -> str:
    if specialist_session_id.startswith("Error:"):
        return f"Cannot delegate: Previous step (start_session_with_specialist) failed: ({specialist_session_id})"
    try:
        print(f"[ClientAgentDefinition Tool] Delegating to {target_agent_name} at {target_agent_execute_url} "
              f"with session {specialist_session_id} and payload: {payload}")
        headers = {"X-Session-Id": specialist_session_id}
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(target_agent_execute_url, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            print(f"[ClientAgentDefinition Tool] Response from {target_agent_name}: {response_data}")
            if response_data.get("final_agent_utterance"):
                return response_data.get("final_agent_utterance")
            elif response_data.get("error_message"):
                return f"Error from {target_agent_name}: {response_data.get('error_message')}"
            else:
                return f"Unexpected response structure from {target_agent_name}: {json.dumps(response_data)}"
    except httpx.HTTPStatusError as e:
        error_body = "No additional error details in response."
        try: error_body = e.response.json()
        except: pass
        print(f"[ClientAgentDefinition Tool] HTTP Error (with session) delegating to {target_agent_name}: {e.response.status_code} - {error_body}")
        return f"Failed to delegate to {target_agent_name} (with session). HTTP Error: {e.response.status_code}. Details: {error_body}"
    except Exception as e:
        print(f"[ClientAgentDefinition Tool] General Error (with session) delegating to {target_agent_name}: {e}")
        return f"Failed to delegate to {target_agent_name} (with session). Error: {str(e)}"


root_agent = Agent(
    name="ClientAgentWithDiscoveryAndPersistentSession",
    model=MODEL_NAME,
    description="Routes queries to specialist HTTP agents, managing its own sessions in a database and sessions with specialists.",
    instruction=_client_agent_instruction, 
    tools=[fetch_agent_card, start_session_with_specialist, delegate_task_with_session]
)

APP_NAME = "ClientAgentDBServiceApp"
USER_ID = "client_agent_persistent_user_001"

try:
    session_service = DatabaseSessionService(db_url=CLIENT_AGENT_DB_URL_FROM_ENV)
    print(f"[ClientAgent Definition - agent.py] DatabaseSessionService initialized for ClientAgent with URL: {CLIENT_AGENT_DB_URL_FROM_ENV}")
except Exception as e:
    print(f"[ClientAgent Definition - agent.py] FAILED to initialize DatabaseSessionService: {e}")
    print("[ClientAgent Definition - agent.py] WARNING: Falling back to InMemorySessionService for ClientAgent due to DB init error.")
    from google.adk.sessions import InMemorySessionService
    session_service = InMemorySessionService()

runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

print(f"[ClientAgent Definition - agent.py] It will attempt to discover agents at: {SPECIALIST_AGENT_DISCOVERY_BASE_URLS}")
print(f"[ClientAgent Definition - agent.py] It will use '{MAIN_API_SERVER_BASE_URL_FOR_SESSIONS}' as specialist_base_api_url for starting sessions.")