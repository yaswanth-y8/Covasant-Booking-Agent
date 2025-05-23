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
else:
    print(f"[ClientAgent Definition - agent.py] Warning: .env file not found at {DOTENV_PATH}.")

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

import google.generativeai as genai
genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-1.5-flash-latest"

CLIENT_AGENT_DB_URL_FROM_ENV = os.getenv(
    "CLIENT_AGENT_ADK_SQL_DB_URL",
    f"sqlite:///{os.path.join(BASE_DIR, 'client_agent_sessions.db')}"
)

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


instruction_template = ( 
    "You are a master routing agent. Your goal is to assist the user by finding the best specialist agent, "
    "establishing a session with it, and then delegating the task. "
    "You have tools: "
    "`fetch_agent_card(agent_discovery_base_url: str)` - to get capabilities of agents. Use the URLs from the known discovery list. "
    "`start_session_with_specialist(specialist_agent_name: str, specialist_base_api_url: str)` - to get a session_id FOR THE SPECIALIST. For `specialist_base_api_url`, you MUST use the value '{param_specialist_api_base_url}'. "
    "`delegate_task_with_session(target_agent_name: str, target_agent_execute_url: str, specialist_session_id: str, payload: dict)` - to make the call using the obtained specialist_session_id. "
    "Known specialist agent DISCOVERY base URLs (for fetch_agent_card tool) are: {param_discovery_urls}.\n"
    "Procedure:\n"
    "1.  User query received. FIRST use `fetch_agent_card` for EACH `agent_discovery_base_url` to get capabilities. Wait for all card responses.\n"
    "2.  Analyze user query and ALL fetched agent cards (`description`, `capabilities_summary`).\n"
    "3.  Identify the BEST specialist agent. From its card, note its `agent_name` and the FULL `endpoints.execute_task.url` (this is your `target_agent_execute_url`).\n"
    "4.  If a suitable specialist is found:\n"
    
    "    a.  Your NEXT tool call MUST be to `start_session_with_specialist`. Provide its `agent_name` (e.g., 'movie_booking_agent') and for the `specialist_base_api_url` argument, you MUST pass the string '{param_specialist_api_base_url}'. You will get a `specialist_session_id` string as the tool response.\n"
    "    b.  If `start_session_with_specialist` returns a session ID (it does not start with 'Error:'):\n"
    "        i.  Prepare the `payload`. This MUST be `{{'query': 'THE_ORIGINAL_USER_QUERY_TEXT'}}`.\n" 
    "        ii. Your NEXT tool call MUST be to `delegate_task_with_session`. Provide the specialist's `agent_name`, the `target_agent_execute_url` (full URL from card), the `specialist_session_id` you just got, and the `payload`.\n"
    "        iii.You will get the specialist's final response string as the output.\n"
    "        iv. THIS result string IS your final answer to the user. Present it clearly.\n"
    "    c.  If `start_session_with_specialist` returns an error string, inform the user you couldn't connect to the specialist (include the error details if helpful).\n"
    "5.  If NO specialist is suitable OR fetching cards fails: Respond directly to the user.\n"
    "**Flow: Fetch Cards -> Start Specialist Session -> Delegate Task -> Final Answer.**"

     "**IMPORTANT:**\n"
    "- For ANY query that is not a simple greeting (like 'hello', 'hi', 'good morning'), you MUST follow the above steps procedure to delegate to ''.\n"
    "- If the user simply says 'hello' or 'hi', you can respond with a polite greeting like 'Hello! How can I help you?' and then for their NEXT query, follow the delegation procedure.\n"
    
)


_client_agent_instruction = instruction_template.format(
    param_specialist_api_base_url=MAIN_API_SERVER_BASE_URL_FOR_SESSIONS,
    param_discovery_urls=', '.join(SPECIALIST_AGENT_DISCOVERY_BASE_URLS) if SPECIALIST_AGENT_DISCOVERY_BASE_URLS else 'None configured'
)


AGENT_CARDS_CACHE: Dict[str, Dict] = {}
SPECIALIST_SESSIONS_CACHE: Dict[str, str] = {}


async def fetch_agent_card(agent_discovery_base_url: str) -> Optional[Dict]:
    if agent_discovery_base_url in AGENT_CARDS_CACHE:
        return AGENT_CARDS_CACHE[agent_discovery_base_url]
    try:
        card_url = f"{agent_discovery_base_url}/.well-known/agent-card"
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(card_url)
            response.raise_for_status()
            card = response.json()
            AGENT_CARDS_CACHE[agent_discovery_base_url] = card
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
            return cached_session_id
        else:
            print(f"[ClientAgentDefinition Tool] Cached session for {specialist_agent_name} was an error, attempting to get new one.")
    try:
        start_session_url = f"{specialist_base_api_url}/start_agent_interaction/"
        payload = {"agent_name": specialist_agent_name}
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(start_session_url, json=payload)
            response.raise_for_status()
            response_data = response.json()
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
        headers = {"X-Session-Id": specialist_session_id}
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(target_agent_execute_url, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
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
    instruction=_client_agent_instruction, # Use the instruction string defined above
    tools=[fetch_agent_card, start_session_with_specialist, delegate_task_with_session]
)

APP_NAME = "ClientAgentDBServiceApp"
USER_ID = "client_agent_persistent_user_001"

try:
    session_service = DatabaseSessionService(db_url=CLIENT_AGENT_DB_URL_FROM_ENV)
except Exception as e:
    print(f"[ClientAgent Definition - agent.py] FAILED to initialize DatabaseSessionService: {e}")
    print("[ClientAgent Definition - agent.py] WARNING: Falling back to InMemorySessionService for ClientAgent due to DB init error.")
    from google.adk.sessions import InMemorySessionService # Import here if fallback needed
    session_service = InMemorySessionService()

runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)
