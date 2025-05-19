import motor.motor_asyncio
import os
import datetime
from typing import Optional, List
from dotenv import load_dotenv

BASE_DIR_DB = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH_DB = os.path.join(BASE_DIR_DB, '.env')
if os.path.exists(DOTENV_PATH_DB):
    load_dotenv(dotenv_path=DOTENV_PATH_DB)

MONGO_DETAILS = os.getenv("MONGO_DETAILS")
MONGO_DATABASE_NAME = "adk_agent_logs"
SESSION_COLLECTION_NAME = "agent_sessions"
INTERACTION_COLLECTION_NAME = "agent_interactions"

db_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
db: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None


async def connect_to_mongo():
    global db_client, db
    if db_client is None:
        print(f"Connecting to MongoDB at {MONGO_DETAILS} …")
        db_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
        db = db_client[MONGO_DATABASE_NAME]
        print("MongoDB connection established.")


async def close_mongo_connection():
    global db_client, db
    if db_client is not None:
        print("Closing MongoDB connection …")
        db_client.close()
        db_client = None
        db = None
        print("MongoDB connection closed.")


async def log_session_start(
    session_id: str,
    user_id: str,
    agent_name: str,
    query_endpoint_returned: str,
):
    if db is None:
        print("Database not initialized – cannot log session start.")
        return

    session_doc = {
        "_id": session_id,
        "session_id": session_id,
        "user_id": user_id,
        "agent_name": agent_name,
        "query_endpoint_returned": query_endpoint_returned,
        "start_time": datetime.datetime.now(datetime.timezone.utc),
    }
    try:
        await db[SESSION_COLLECTION_NAME].insert_one(session_doc)
        print(f"Session start logged ({session_id}).")
    except Exception as e:
        print(f"Error logging session start: {e}")


async def log_agent_interaction(
    session_id: str,
    user_id: str,
    agent_name: str,
    query: str,
    full_log: List[str],
    final_agent_utterance: Optional[str],
    error: Optional[str],
    turn_start_time: datetime.datetime,
):
    if db is None:
        print("Database not initialized – cannot log interaction.")
        return

    interaction_doc = {
        "session_id": session_id,
        "user_id": user_id,
        "agent_name": agent_name,
        "query": query,
        "full_log": full_log,
        "final_agent_utterance": final_agent_utterance,
        "error": error,
        "timestamp": turn_start_time,
        "turn_duration_ms": (
            datetime.datetime.now(datetime.timezone.utc) - turn_start_time
        ).total_seconds()
        * 1000,
    }
    try:
        await db[INTERACTION_COLLECTION_NAME].insert_one(interaction_doc)
        print(f"Interaction logged ({session_id}).")
    except Exception as e:
        print(f"Error logging interaction: {e}")


def get_db() -> Optional[motor.motor_asyncio.AsyncIOMotorDatabase]:
    return db
