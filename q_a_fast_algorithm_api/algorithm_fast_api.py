from fastapi import FastAPI
from pydantic import BaseModel
from file3 import handle_user_input

app = FastAPI()

current_context = None
history = []

class UserRequest(BaseModel):
    text: str

@app.get("/ask")
async def ask_question(req: UserRequest):
    global current_context, history
    response, current_context, history = handle_user_input(req.text, current_context, history)
    return response