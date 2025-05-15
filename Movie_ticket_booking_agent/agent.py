from typing import Dict, List, Any
from google.adk.agents import Agent
import google.generativeai as genai
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def find_movie_showtimes(movie: str, location: str, date: str) -> dict:
    print(f"[Tool Call] find_movie_showtimes(movie='{movie}', location='{location}', date='{date}')")
    if movie.lower() == "avengers: endgame" and location.lower() == "hyderabad" and date == "2025-05-15":
        return {
            "status": "success",
            "showtimes": ["14:00", "17:30", "21:00"],
        }
    else:
        return {
            "status": "error",
            "error_message": f"No showtimes found for '{movie}' in '{location}' on '{date}'.",
        }

async def select_seats(showtime: str, num_seats: int, preferences: str = "") -> Dict[str, Any]:
    print(f"[Tool Call] select_seats(showtime='{showtime}', num_seats={num_seats}, preferences='{preferences}')")
    if num_seats <= 2:
        selected_seats_list = ["A5", "A6"] if num_seats == 2 else ["B3"]
        return {
            "status": "success",
            "seats": selected_seats_list,
            "message": f"Selected {num_seats} seats ({', '.join(selected_seats_list)}) for {showtime} (preferences: {preferences if preferences else 'none'}).",
        }
    else:
        return {
            "status": "error",
            "error_message": f"Could not select {num_seats} seats for {showtime}. Maximum 2 seats allowed in this demo.",
        }

async def confirm_booking(movie: str, showtime: str, seats: List[str]) -> dict:
    print(f"[Tool Call] confirm_booking(movie='{movie}', showtime='{showtime}', seats={seats})")
    booking_id = "BOOKING12345"
    return {
        "status": "success",
        "booking_id": booking_id,
        "confirmation_message": (
            f"Your booking for '{movie}' at {showtime} in seats {', '.join(seats)} is confirmed."
            f" Your booking ID is {booking_id}."
        ),
    }

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
genai.configure(api_key=api_key)

MODEL_NAME = "gemini-1.5-flash-latest"

root_agent = Agent(
    name="movie_ticket_agent",
    model=MODEL_NAME,
    description="Agent to help users book movie tickets by finding showtimes, selecting seats, and confirming bookings.",
    instruction=(
        "You are a helpful and friendly assistant for booking movie tickets. "
        "Use the available tools to find showtimes, select seats, and confirm bookings based on the user's requests. "
        "Ensure you gather all necessary information for each step. For example, to find showtimes, you need the movie, location, and date. "
        "For seat selection, you need the showtime and number of seats. For booking confirmation, you need movie, showtime, and the list of selected seats."
    ),
    tools=[find_movie_showtimes, select_seats, confirm_booking],
)

async def main():
    print(f"--- Starting Movie Ticket Booking Agent (Model: {MODEL_NAME}) ---\n")
    query1 = "Find showtimes for Avengers: Endgame in Hyderabad on May 15, 2025."
    print(f"User: {query1}")
    response_showtimes = await root_agent.run(query=query1)
    print(f"Agent: {response_showtimes}\n")

    query2 = "Great, I'd like to select 2 seats for the 14:00 showtime. No preferences."
    print(f"User: {query2}")
    response_seats = await root_agent.run(query=query2)
    print(f"Agent: {response_seats}\n")

    query3 = "Okay, please confirm my booking for Avengers: Endgame at 14:00 in seats A5 and A6."
    print(f"User: {query3}")
    response_confirmation = await root_agent.run(query=query3)
    print(f"Agent: {response_confirmation}\n")
    print("--- End of Interaction ---")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ImportError as e:
        if "google.adk" in str(e):
            print("Error: The 'google.adk' package is not installed or available.")
            print("This example requires the Google Agent Development Kit (ADK).")
            print("Please ensure it's installed and you have access to it.")
        else:
            print(f"An import error occurred: {e}")
            print("Please ensure all dependencies like 'google-generativeai' and 'python-dotenv' are installed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
