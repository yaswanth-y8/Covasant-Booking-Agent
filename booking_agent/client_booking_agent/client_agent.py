# -*- coding: utf-8 -*-
# client_agent_definition.py

import os
import json
from dotenv import load_dotenv
from typing import Dict, Any, Tuple # Tuple for returning session_id and endpoint
import httpx

from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found (client_agent_definition).")

import google.generativeai as genai
genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-1.5-flash-latest"
print(f"[ClientAgentDefinition] Using Google Model: {MODEL_NAME}")

# Base URLs for agent *card discovery*. These should point to the root of where agent-card is.
# e.g., if movie agent card is at http://localhost:8000/movie_booking_agent/.well-known/agent-card
# then its base discovery URL is http://localhost:8000/movie_booking_agent
SPECIALIST_AGENT_DISCOVERY_BASE_URLS = []
movie_agent_discovery_base = os.getenv("MOVIE_BOOKING_AGENT_DISCOVERY_BASE_URL") # e.g., http://localhost:8000/movie_booking_agent
if movie_agent_discovery_base:
    SPECIALIST_AGENT_DISCOVERY_BASE_URLS.append(movie_agent_discovery_base)

bus_agent_discovery_base = os.getenv("BUS_BOOKING_AGENT_DISCOVERY_BASE_URL")   # e.g., http://localhost:8000/bus_booking_agent
if bus_agent_discovery_base:
    SPECIALIST_AGENT_DISCOVERY_BASE_URLS.append(bus_agent_discovery_base)

if not SPECIALIST_AGENT_DISCOVERY_BASE_URLS:
    print("[ClientAgentDefinition] WARNING: No specialist agent discovery base URLs configured. ClientAgent may not find agents.")

# Cache for fetched agent cards
AGENT_CARDS_CACHE = {}
# Cache for active sessions with specialist agents (session_id_for_specialist: str)
# Key: target_agent_name, Value: session_id obtained from specialist
SPECIALIST_SESSIONS_CACHE: Dict[str, str] = {}


async def fetch_agent_card(agent_discovery_base_url: str) -> Dict | None:
    # ... (fetch_agent_card function remains the same as your working version)
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
    specialist_agent_name: str,      # e.g., "movie_booking_agent"
    specialist_base_api_url: str     # e.g., "http://localhost:8000" (where main_api.py runs)
) -> str: # Returns session_id from specialist or error string
    """
    Calls the specialist's /start_agent_interaction/ endpoint to get a session ID.
    Caches the session ID.
    """
    if specialist_agent_name in SPECIALIST_SESSIONS_CACHE:
        print(f"[ClientAgentDefinition Tool] Using cached session for {specialist_agent_name}: {SPECIALIST_SESSIONS_CACHE[specialist_agent_name]}")
        return SPECIALIST_SESSIONS_CACHE[specialist_agent_name]

    try:
        # Your main_api.py /start_agent_interaction expects {'agent_name': '...'}
        start_session_url = f"{specialist_base_api_url}/start_agent_interaction/"
        payload = {"agent_name": specialist_agent_name}
        print(f"[ClientAgentDefinition Tool] Starting session with {specialist_agent_name} at {start_session_url}...")
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
                return f"Error: Could not get session_id from {specialist_agent_name}. Response: {response_data}"
    except Exception as e:
        print(f"[ClientAgentDefinition Tool] Error starting session with {specialist_agent_name}: {e}")
        return f"Error: Failed to start session with {specialist_agent_name}: {str(e)}"


async def delegate_task_with_session(
    target_agent_name: str,                 # For logging
    target_agent_execute_url: str,          # FULL URL for /query endpoint from card
    specialist_session_id: str,             # Session ID for the specialist
    payload: Dict[str, Any]                 # e.g., {"query": "user's actual query"}
) -> str:
    """Delegates task using an existing session_id for the specialist."""
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


_client_agent_instruction = (
    "You are a master routing agent. Your goal is to assist the user by finding the best specialist agent, "
    "establishing a session with it, and then delegating the task. "
    "You have tools: "
    "`fetch_agent_card(agent_discovery_base_url: str)` - to get capabilities of agents. "
    "`start_session_with_specialist(specialist_agent_name: str, specialist_base_api_url: str)` - to get a session_id FOR THE SPECIALIST. The `specialist_base_api_url` is the root URL of the API serving that specialist (e.g., http://localhost:8000). "
    "`delegate_task_with_session(target_agent_name: str, target_agent_execute_url: str, specialist_session_id: str, payload: dict)` - to make the call using the obtained specialist_session_id. "
    f"Known specialist agent DISCOVERY base URLs are: {', '.join(SPECIALIST_AGENT_DISCOVERY_BASE_URLS) if SPECIALIST_AGENT_DISCOVERY_BASE_URLS else 'None configured'}.\n"
    "Procedure:\n"
    "1.  User query received. FIRST use `fetch_agent_card` for EACH `agent_discovery_base_url` to get capabilities. Wait for all card responses.\n"
    "2.  Analyze user query and ALL fetched agent cards (`description`, `capabilities_summary`).\n"
    "3.  Identify the BEST specialist agent. From its card, note its `agent_name` and the FULL `endpoints.execute_task.url` (this is your `target_agent_execute_url`). Also note the general `specialist_base_api_url` that serves this agent (this will be one of the initial discovery URLs, e.g., http://localhost:8000 if multiple agents are served from there).\n"
    "4.  If a suitable specialist is found:\n"
    "    a.  Your NEXT tool call MUST be to `start_session_with_specialist`. Provide its `agent_name` (e.g., 'movie_booking_agent') and the `specialist_base_api_url` where it is hosted (e.g., 'http://localhost:8000'). You will get a `specialist_session_id` string as the tool response (or an error).\n"
    "    b.  If you successfully get a `specialist_session_id`:\n"
    "        i.  Prepare the `payload` for the task. This MUST be `{{'query': 'THE_ORIGINAL_USER_QUERY_TEXT'}}`.\n"
    "        ii. Your NEXT tool call MUST be to `delegate_task_with_session`. Provide the specialist's `agent_name`, the `target_agent_execute_url` (full URL from its card), the `specialist_session_id` you just got, and the `payload`.\n"
    "        iii.You will get the specialist's final response string as the output of `delegate_task_with_session`.\n"
    "        iv. THIS result string IS your final answer to the user. Present it clearly.\n"
    "    c.  If starting the session fails, inform the user you couldn't connect to the specialist.\n"
    "5.  If NO specialist is suitable OR fetching cards fails for all: Respond directly to the user.\n"
    "**Flow: Fetch Cards -> Start Specialist Session -> Delegate Task with Session -> Final Answer.**"
)

client_agent = Agent( # Renamed to be less generic, e.g. 'client_discovery_agent_def'
    name="ClientAgentWithHttpDiscoveryAndSessionMgmt",
    model=MODEL_NAME,
    description="Routes queries to specialist HTTP agents, managing sessions with them.",
    instruction=_client_agent_instruction,
    tools=[fetch_agent_card, start_session_with_specialist, delegate_task_with_session]
)

APP_NAME = "ClientAgentHttpServiceAppDefinition" # Default AppName
USER_ID = "client_agent_def_user_001"      # Default UserID for this agent definition context

# Default runner for potential standalone testing of client_agent_definition.py logic
_default_session_service = InMemorySessionService()
runner = Runner( # This runner instance might not be used by the service if it creates its own
    agent=client_agent,
    app_name=APP_NAME,
    session_service=_default_session_service,
)

print(f"[ClientAgentDefinition] ADK ClientAgent '{client_agent.name}' definition complete.")
print(f"[ClientAgentDefinition] (Note: Service main will create its own Runner with specified session service)")