from fastapi import FastAPI
from . import  agent 
app=FastAPI()



@app.get("/{agentname}")
def get_agent_name(agentname:str):
    _agent=""
    if agentname=="movie_ticket_agent":
        _agent =agent
    return _agent.start()