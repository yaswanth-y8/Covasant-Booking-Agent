import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route, Mount
import datetime
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from typing import Dict, List, Any 

mcp = FastMCP("booking_services_mcp") 


@mcp.tool()
async def find_available_buses(origin: str, destination: str, travel_date: str) -> dict:
    """Finds available buses based on origin, destination, and travel date.

    Args:
        origin (str): The starting city or location for the bus journey.
        destination (str): The ending city or location for the bus journey.
        travel_date (str): The date of travel in YYYY-MM-DD format.
    """
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

@mcp.tool()
async def select_bus_and_seats(bus_id: str, num_seats_to_book: int, seat_preferences: str = "") -> dict:
    """Selects a specific bus and number of seats, optionally with preferences.

    Args:
        bus_id (str): The ID of the bus selected from the search results.
        num_seats_to_book (int): The number of seats the user wants to book.
        seat_preferences (str, optional): User's preferences for seats (e.g., 'window', 'aisle', 'front'). Defaults to an empty string.
    """
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

@mcp.tool()
async def confirm_bus_booking(bus_id: str, passenger_name: str, passenger_contact: str, seats_booked: List[str]) -> dict:
    """Confirms a bus booking with passenger details for already selected seats.

    Args:
        bus_id (str): The ID of the bus for which seats were provisionally held.
        passenger_name (str): The full name of the primary passenger.
        passenger_contact (str): The contact number (e.g., mobile phone) of the primary passenger.
        seats_booked (List[str]): A list of seat identifiers that were provisionally selected (e.g., ['W1', 'W2']).
    """
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

@mcp.tool()
async def find_movie_showtimes(movie: str, location: str, date: str) -> dict:
    """Finds available showtimes for a specific movie in a given location and date.

    Args:
        movie (str): The name of the movie to search for.
        location (str): The city or area where the user wants to watch the movie.
        date (str): The desired date for watching the movie, in YYYY-MM-DD format.
    """
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
@mcp.tool()
async def select_seats(showtime: str, num_seats: int, preferences: str = "") -> Dict[str, Any]:
    """Selects a specified number of seats for a given movie showtime, with optional preferences.

    Args:
        showtime (str): The specific showtime selected by the user (e.g., '14:00').
        num_seats (int): The number of seats the user wishes to book.
        preferences (str, optional): User's preferences for seating (e.g., 'front row', 'near aisle'). Defaults to an empty string.
    """
    print(f"[Tool Call] select_seats(showtime='{showtime}', num_seats={num_seats}, preferences='{preferences}')")
    if num_seats <= 2: # Assuming this logic applies to movie seats as well for the demo
        selected_seats_list = ["A5", "A6"] if num_seats == 2 else ["B3"]
        return {
            "status": "success",
            "seats": selected_seats_list,
            "message": f"Selected {num_seats} seats ({', '.join(selected_seats_list)}) for the {showtime} showtime (preferences: {preferences if preferences else 'none'}).",
        }
    else:
        return {
            "status": "error",
            "error_message": f"Could not select {num_seats} seats for showtime {showtime}. Maximum 2 seats allowed in this demo for movies as well.",
        }
@mcp.tool()
async def confirm_booking(movie: str, showtime: str, seats: List[str]) -> dict:
    """Confirms a movie ticket booking for the selected movie, showtime, and seats.

    Args:
        movie (str): The name of the movie being booked.
        showtime (str): The showtime for which the booking is being made.
        seats (List[str]): A list of seat identifiers that have been selected (e.g., ['A5', 'A6']).
    """
    # Note: This function signature for movie confirm_booking is different from bus_confirm_booking.
    # It doesn't take passenger details here, which might be a simplification for this example.
    # If passenger details are needed, the signature should be updated.
    print(f"[Tool Call] confirm_booking(movie='{movie}', showtime='{showtime}', seats={seats})")
    booking_id = f"MOVIE_BOOKING_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}" # More unique movie booking ID
    return {
        "status": "success",
        "booking_id": booking_id,
        "confirmation_message": (
            f"Your booking for '{movie}' at {showtime} in seats {', '.join(seats)} is confirmed."
            f" Your booking ID is {booking_id}."
        ),
    }



sse = SseServerTransport("/messages/")

async def handle_sse(request: Request) -> None:
    _server = mcp._mcp_server
    async with sse.connect_sse(
        request.scope,
        request.receive,
        request._send,
    ) as (reader, writer):
        await _server.run(reader, writer, _server.create_initialization_options())

app = Starlette(
    debug=True,
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ],
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8005)

