from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams


agent_name="bus_booking_agent"
async def create_root():
    tools,exit_stack = await MCPToolset.from_server(connection_params=SseServerParams(url="http://localhost:8005/sse"))

    agent = LlmAgent(
        model="gemini-2.0-flash",
        name=agent_name,
        description="Agent to help users find and book bus tickets.",
        instruction=(
        "You are a helpful assistant for booking bus tickets. Your tasks are:\n"
        "1. Find available buses: When a user wants to find buses, use the 'find_available_buses' tool. You'll need the origin city, destination city, and travel date (YYYY-MM-DD).\n"
        "2. Select bus and seats: After buses are found and the user chooses one, use the 'select_bus_and_seats' tool. You'll need the bus ID and the number of seats. Seat preferences are optional.\n"
        "3. Confirm booking: To finalize the booking, use the 'confirm_bus_booking' tool. You'll need the bus ID, primary passenger's name, passenger's contact number, and the list of seats that were selected.\n"
        "Always ask for any missing information before calling a tool. Provide clear summaries of tool outputs."
    ),
        tools=tools,
    )
    print(agent)
    return agent,exit_stack

root_agent = create_root()
