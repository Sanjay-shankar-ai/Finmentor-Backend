
from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from new import handle_query
import uvicorn
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask(
    
    request: Request,
    question: str = Form(...),
    user_id: str = Form("demo-user"),
    use_case: str = Form(None)
):
    print("\n✅ POST /ask called")
    print(f"📩 question = {question}")
    print(f"👤 user_id = {user_id}")
    print(f"🧠 use_case = {use_case}")
    start_time = time.time()
    
    try:
        response = handle_query(question, user_id=user_id)
        duration = time.time() - start_time
        print(f"⏱️ Response time: {duration:.2f} seconds")
        print("✅ Response generated successfully.")
        return {"response": response}
    
    except Exception as e:
        print("❌ Error in handle_query:", str(e))
        return {"response": "Internal server error. Please check backend logs."}


if __name__ == "__main__":
    print("🚀 Starting FastAPI server...")
    uvicorn.run("chatbot_api:app", host="0.0.0.0", port=8000, reload=True)
