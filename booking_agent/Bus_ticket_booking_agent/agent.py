from typing import Dict, List, Any
from google.adk.agents import Agent
import google.generativeai as genai
import os
from dotenv import load_dotenv
import asyncio
import datetime 

load_dotenv()

async def find_available_buses(origin: str, destination: str, travel_date: str) -> dict:
    print(f"[Tool Call] find_available_buses(origin='{origin}', destination='{destination}', travel_date='{travel_date}')")
    if origin.lower() == "mumbai" and destination.lower() == "pune" and travel_date == "2025-07-20":
        return {
            "status": "success",
            "buses": [
                {"bus_id": "BUS789", "operator_name": "Red Travels", "departure_time": "09:00", "arrival_time": "13:00", "price_per_seat": 500, "available_seats": 15},
                {"bus_id": "BUS123", "operator_name": "BlueLine Express", "departure_time": "11:30", "arrival_time": "15:30", "price_per_seat": 550, "available_seats": 10},
            ],
        }
    elif origin.lower() == "delhi" and destination.lower() == "jaipur" and travel_date == "2025-08-10":
        return {
            "status": "success",
            "buses": [
                {"bus_id": "BUS456", "operator_name": "Green Ways", "departure_time": "07:00", "arrival_time": "12:00", "price_per_seat": 700, "available_seats": 20},
            ],
        }
    else:
        return {
            "status": "error",
            "error_message": f"No buses found from '{origin}' to '{destination}' on '{travel_date}'.",
        }

async def select_bus_and_seats(bus_id: str, num_seats_to_book: int, seat_preferences: str = "") -> dict:
    print(f"[Tool Call] select_bus_and_seats(bus_id='{bus_id}', num_seats_to_book={num_seats_to_book}, seat_preferences='{seat_preferences}')")
    if bus_id == "BUS789" and num_seats_to_book <= 2:
        provisional_seats = ["W1", "W2"] if num_seats_to_book == 2 else ["W1"]
        total_price = num_seats_to_book * 500
        return {
            "status": "success",
            "provisional_seats": provisional_seats,
            "total_price": total_price,
            "message": f"{num_seats_to_book} seats ({', '.join(provisional_seats)}) provisionally held on bus {bus_id}. Preferences: {seat_preferences if seat_preferences else 'none'}.",
        }
    elif bus_id == "BUS456" and num_seats_to_book == 1:
        return {
            "status": "success",
            "provisional_seats": ["F3"],
            "total_price": 700,
            "message": f"1 seat (F3) provisionally held on bus {bus_id}. Preferences: {seat_preferences if seat_preferences else 'none'}.",
        }
    else:
        return {
            "status": "error",
            "error_message": f"Could not select {num_seats_to_book} seats on bus '{bus_id}'. They might be unavailable or the bus ID is invalid.",
        }

async def confirm_bus_booking(bus_id: str, passenger_name: str, passenger_contact: str, seats_booked: List[str]) -> dict:
    print(f"[Tool Call] confirm_bus_booking(bus_id='{bus_id}', passenger_name='{passenger_name}', passenger_contact='{passenger_contact}', seats_booked={seats_booked})")
    if passenger_name and passenger_contact and seats_booked:
        pnr_number = f"PNR{bus_id.replace('BUS', '')}{datetime.datetime.now().strftime('%H%M%S')}"
        return {
            "status": "success",
            "pnr_number": pnr_number,
            "message": f"Booking confirmed for {passenger_name} on bus {bus_id} for seats {', '.join(seats_booked)}. PNR: {pnr_number}. Contact: {passenger_contact}",
        }
    else:
        return {
            "status": "error",
            "error_message": "Booking confirmation failed. Missing passenger details or seats.",
        }

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
genai.configure(api_key=api_key)



root_agent = Agent(
    name="bus_ticket_booking_agent",
    model=os.getenv("MODEL_NAME"),
    description="Agent to help users find and book bus tickets.",
    instruction=(
        "You are a helpful assistant for booking bus tickets. Your tasks are:\n"
        "1. Find available buses: When a user wants to find buses, use the 'find_available_buses' tool. You'll need the origin city, destination city, and travel date (YYYY-MM-DD).\n"
        "2. Select bus and seats: After buses are found and the user chooses one, use the 'select_bus_and_seats' tool. You'll need the bus ID and the number of seats. Seat preferences are optional.\n"
        "3. Confirm booking: To finalize the booking, use the 'confirm_bus_booking' tool. You'll need the bus ID, primary passenger's name, passenger's contact number, and the list of seats that were selected.\n"
        "Always ask for any missing information before calling a tool. Provide clear summaries of tool outputs."
    ),
    tools=[find_available_buses, select_bus_and_seats, confirm_bus_booking],
)

# async def main():
    # print(f"--- Starting Bus Ticket Booking Agent (Model: {MODEL_NAME}) ---\n")

    # query1 = "I want to find buses from Mumbai to Pune for July 20, 2025."
    # print(f"User: {query1}")
    # response1 = await root_agent.run(query=query1)
    # print(f"Agent: {response1}\n")

    # query2 = "Okay, I'd like to book 2 seats on BUS789. Window side preferred."
    # print(f"User: {query2}")
    # response2 = await root_agent.run(query=query2)
    # print(f"Agent: {response2}\n")

    # query3 = "Please confirm my booking for BUS789. Passenger name is Anand Kumar, contact is 9988776655, and seats are W1, W2."
    # print(f"User: {query3}")
    # response3 = await root_agent.run(query=query3)
    # print(f"Agent: {response3}\n")

    # query4 = "Are there any buses from Delhi to Jaipur on August 10, 2025?"
    # print(f"User: {query4}")
    # response4 = await root_agent.run(query=query4)
    # print(f"Agent: {response4}\n")

    # print("--- End of Interaction ---")

# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except ImportError as e:
#         if "google.adk" in str(e):
#             print("Error: The 'google.adk' package is not installed or available.")
#             print("This example requires the Google Agent Development Kit (ADK).")
#             print("Please ensure it's installed and you have access to it.")
#         else:
#             print(f"An import error occurred: {e}")
#             print("Please ensure all dependencies like 'google-generativeai' and 'python-dotenv' are installed.")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
