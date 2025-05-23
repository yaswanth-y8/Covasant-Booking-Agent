from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams

async def create_root():
    tools,exit_stack = await MCPToolset.from_server(connection_params=SseServerParams(url="http://localhost:8005/sse"))

    agent = LlmAgent(
        model="gemini-2.0-flash",
        name="movie_ticket_agent",
        description="Agent to help users book movie tickets by finding showtimes, selecting seats, and confirming bookings.",
    instruction=(
        "You are a helpful and friendly assistant for booking movie tickets. "
        "Use the available tools to find showtimes, select seats, and confirm bookings based on the user's requests. "
        "Ensure you gather all necessary information for each step. For example, to find showtimes, you need the movie, location, and date. "
        "if user passes the date in different format change the format to yyyy-mm-dd before sending it to tool"
        "For seat selection, you need the showtime and number of seats. For booking confirmation, you need movie, showtime, and the list of selected seats."
    ),
        tools=tools,
    )
    return agent,exit_stack

root_agent = create_root()
