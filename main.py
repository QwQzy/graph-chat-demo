from uuid import uuid4
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request,Form
from fastapi.responses import HTMLResponse, FileResponse,JSONResponse
from langchain.messages import HumanMessage
from sse_starlette.sse import EventSourceResponse
from motor.motor_asyncio import AsyncIOMotorCollection

from agent.agent_builder import create_chat_agent,CompiledStateGraph,create_abstract_agent
from database.mongodb_client import get_mongo_client

agent:CompiledStateGraph = None
abstract_agent:CompiledStateGraph = None
session_collection:AsyncIOMotorCollection = None

@asynccontextmanager
async def lifespan(app:FastAPI):
    global agent,session_collection,abstract_agent

    mongo_client = get_mongo_client()
    session_collection = mongo_client.get_database("checkpointing_db").get_collection("sessions")

    agent = create_chat_agent(mongo_client)
    abstract_agent = create_abstract_agent()

    try:
        yield
    finally:
        mongo_client.close()

app = FastAPI(lifespan=lifespan)
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return FileResponse("index.html")

@app.post("/stream_chat")
async def chat(request: Request,message:str=Form(...)):
    user_id = request.headers.get("x-user")
    sess_id = request.headers.get("x-sess")
    if not user_id or not sess_id:
        return JSONResponse({"code":-1,"error":"invalid user"})

    if not message:
        return JSONResponse({"code":-1,"error":"invalid message"})

    async def event_stream():
        # sse通信
        async for chunk in agent.astream_events(
                {"messages": [HumanMessage(content=message)]},
                config={
                    "configurable": {
                        "thread_id": f"{user_id}-{sess_id}",
                    }
                }
        ):
            if chunk["event"] == "on_chat_model_stream":
                if chunk["data"]["chunk"].content:
                    yield {"data":chunk["data"]["chunk"].content}

    return EventSourceResponse(event_stream())

@app.post("/create_session")
async def create_session(request: Request):
    user_id = request.headers.get("x-user")
    if not user_id:
        return JSONResponse({"code":-1,"error":"invalid user"})

    doc = {
        "user_id": user_id,
        "session_id": str(uuid4()),
        "name":"新会话",
    }
    await session_collection.insert_one(doc)

    doc.pop("_id")

    return JSONResponse({"code":0,"data":doc})

@app.post("/update_session")
async def update_session(request: Request,name:str=Form(...)):
    user_id = request.headers.get("x-user")
    sess_id = request.headers.get("x-sess")
    if not user_id or not sess_id:
        return JSONResponse({"code": -1, "error": "invalid user"})

    if not name:
        name = "新会话"

    await session_collection.update_one(
        {"user_id":user_id,"session_id":sess_id},
        {"$set":{"name":name}}
    )

    return JSONResponse({"code":0,"data":{"name":name}})

@app.post("/abstract_session")
async def abstract_session(request: Request,first_message:str=Form(...)):
    user_id = request.headers.get("x-user")
    if not user_id:
        return JSONResponse({"code": -1, "error": "invalid user"})

    if not first_message:
        return JSONResponse({"code": -1, "error": "invalid message"})

    async def event_stream():
        # sse通信
        async for chunk in abstract_agent.astream_events({"messages":[first_message]}):
            if chunk["event"] == "on_chat_model_stream":
                if chunk["data"]["chunk"].content:
                    yield {"data": chunk["data"]["chunk"].content}

    return EventSourceResponse(event_stream())


@app.post("/delete_session")
async def delete_session(request: Request):
    user_id = request.headers.get("x-user")
    sess_id = request.headers.get("x-sess")
    if not user_id or not sess_id:
        return JSONResponse({"code": -1, "error": "invalid user"})

    await session_collection.delete_one({"user_id":user_id,"session_id":sess_id})

    return JSONResponse({"code":0,"data":{"session_id":sess_id}})

@app.get("/list_session")
async def list_session(request: Request):
    user_id = request.headers.get("x-user")
    if not user_id:
        return JSONResponse({"code": -1, "error": "invalid user"})

    data = []
    async for record in session_collection.find({"user_id":user_id},projection={"_id":False}):
        data.append(record)

    return JSONResponse({"code":0,"data":data})

@app.get("/list_history_message")
async def list_history_message(request: Request):
    user_id = request.headers.get("x-user")
    sess_id = request.headers.get("x-sess")
    if not user_id or not sess_id:
        return JSONResponse({"code": -1, "error": "invalid user"})

    messages = []
    async for chunk in agent.aget_state_history(
            config={
                "configurable": {
                    "thread_id": f"{user_id}-{sess_id}",
                }
            }
    ):
        messages = chunk.values.get("messages",[])
        break

    return [{"role":msg.type,"content":msg.content} for msg in messages]



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
