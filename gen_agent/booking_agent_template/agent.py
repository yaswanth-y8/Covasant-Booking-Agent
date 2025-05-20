
from typing import Dict, List, Any, Callable, Coroutine
from google.adk.agents import Agent
import google.generativeai as genai
import os
from dotenv import load_dotenv
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types as genai_types

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
genai.configure(api_key=API_KEY)

async def find_movie_showtimes(movie: str, location: str, date: str) -> dict:
    if movie.lower() == "avengers: endgame" and location.lower() == "hyderabad" and date == "2025-05-15":
        return {"status": "success", "showtimes": ["14:00", "17:30", "21:00"]}
    else:
        return {"status": "error", "error_message": f"No showtimes found for '{movie}' in '{location}' on '{date}'."}

async def select_seats(showtime: str, num_seats: int, preferences: str = "") -> Dict[str, Any]:
    if num_seats <= 2:
        selected_seats_list = ["A5", "A6"] if num_seats == 2 else ["B3"]
        return {"status": "success", "seats": selected_seats_list, "message": f"Selected {num_seats} seats ({', '.join(selected_seats_list)}) for {showtime}."}
    else:
        return {"status": "error", "error_message": f"Could not select {num_seats} seats. Max 2 allowed."}

async def confirm_booking(movie: str, showtime: str, seats: List[str]) -> dict:
    booking_id = "BOOKING12345"
    return {"status": "success", "booking_id": booking_id, "confirmation_message": f"Booking for '{movie}' at {showtime} in seats {', '.join(seats)} confirmed. ID: {booking_id}."}

class AgentTemplate:
    USER_ID = "user_1"
    def __init__(self, model_name: str, agent_name: str, agent_description: str, agent_instruction: str, tools: List[Callable[..., Coroutine[Any, Any, Dict]]] = None):
        self.model_name = model_name
        self.agent_name = agent_name 
        self.agent_description = agent_description
        self.agent_instruction = agent_instruction
        self.tools = tools if tools is not None else []
        self.root_agent = Agent(name=self.agent_name, model=self.model_name, description=self.agent_description, instruction=self.agent_instruction, tools=self.tools)
        self.session_service = InMemorySessionService()
        session = self.session_service.create_session(app_name=self.agent_name, user_id=self.USER_ID)
        self.session_id = session.id 
        self.runner = Runner(agent=self.root_agent, app_name=self.agent_name, session_service=self.session_service)

    async def call_agent_async(self, query: str) -> str:
        content = genai_types.Content(role='user', parts=[genai_types.Part(text=query)])
        final_response_text = "Agent did not produce a final response."
        async for event in self.runner.run_async(user_id=self.USER_ID, session_id=self.session_id, new_message=content):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                break
        return final_response_text

    def get_session_details(self) -> Dict[str, str]:
        return {"agent_name": self.agent_name, "session_id": self.session_id, "description": self.agent_description}

ACTIVE_AGENT_SESSIONS: Dict[str, AgentTemplate] = {}
DEFAULT_TOOLS = [find_movie_showtimes, select_seats, confirm_booking]

def create_new_agent_session_instance(model_name: str, agent_name: str, agent_description: str, agent_instruction: str) -> AgentTemplate:
    new_agent_instance = AgentTemplate(
        model_name=model_name,
        agent_name=agent_name, 
        agent_description=agent_description,
        agent_instruction=agent_instruction,
        tools=DEFAULT_TOOLS
    )
    if new_agent_instance.session_id in ACTIVE_AGENT_SESSIONS:
        raise ValueError(f"Session ID {new_agent_instance.session_id} collision. This should not happen.")
    ACTIVE_AGENT_SESSIONS[new_agent_instance.session_id] = new_agent_instance
    return new_agent_instance

def get_agent_by_session_id(session_id: str) -> AgentTemplate | None:
    return ACTIVE_AGENT_SESSIONS.get(session_id)

def list_all_active_session_details() -> List[Dict[str, str]]:
    return [instance.get_session_details() for instance in ACTIVE_AGENT_SESSIONS.values()]