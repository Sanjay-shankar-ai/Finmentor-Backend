from flask import Flask, request, redirect
import requests
from threading import Thread

app = Flask(__name__)

VERIFY_TOKEN = "@Naveen123"
ACCESS_TOKEN = "EAAIvckoIiasBPMGmwhtcAZCNvEIlH6Eq5ct4m2n5zMK7vW6PvWIBMfRFCvwxErKNZBKhzPmAr1bPqka6ZCY7VrCZAIMYDpkhPYCAodOq5nRN6AwKH7bTdB0NsCia81R6Qm4uGG2NZCZBEnpldrE925eM73tnTVnna9C6P0RNCVVqufWDCcaNqSXSb8BcT7"
AI_MODEL_API = "http://localhost:8000/ask"

processed_message_ids = set()

@app.route("/webhook", methods=["GET"])
def verify():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("✅ Webhook verified by Meta.")
        return challenge, 200

    print("❌ Webhook verification failed.")
    return "Verification failed", 403

@app.route("/webhook", methods=["POST"])
def receive_message():
    data = request.get_json()
    print("\n📥 Received Webhook Data:", data)

    def process():
        try:
            for entry in data.get("entry", []):
                for messaging in entry.get("messaging", []):
                    sender_id = messaging.get("sender", {}).get("id")
                    message_text = messaging.get("message", {}).get("text")
                    message_id = messaging.get("message", {}).get("mid", "")

                    if not sender_id or not message_text:
                        return

                    if message_id in processed_message_ids:
                        print(f"🔁 Duplicate message ID {message_id} — Skipping.")
                        return
                    processed_message_ids.add(message_id)

                    print(f"📨 From {sender_id}: {message_text}")

                    try:
                        ai_response = requests.post(AI_MODEL_API, data={
                            "question": message_text,
                            "user_id": "6fbf1e44-0a13-4e59-8eb6-303a9a9be8b0",
                        })
                        if ai_response.status_code == 200:
                            model_reply = ai_response.json().get("response", "⚠ Empty response.")
                        else:
                            model_reply = "⚠ Model error"
                    except Exception as e:
                        print("❌ Error calling model:", e)
                        model_reply = "⚠ AI unavailable."

                    print(f"🤖 Reply: {model_reply}")

                    payload = {
                        "messaging_product": "instagram",
                        "recipient": {"id": sender_id},
                        "message": {"text": model_reply}
                        }

                    headers = {
                        "Authorization": f"Bearer {ACCESS_TOKEN}",
                        "Content-Type": "application/json"
                    }
                    INSTAGRAM_ID = "17841475593690110"
                    send_url = f"https://graph.facebook.com/v18.0/{INSTAGRAM_ID}/me/messages"

                    response = requests.post(send_url, headers=headers, json=payload)
                    print(f"📤 Instagram send status: {response.status_code} {response.text}")

        except Exception as e:
            print("❌ Error:", e)

    Thread(target=process).start()
    return "ok", 200

# OAuth Callback
@app.route("/instagram/callback")
def instagram_callback():
    code = request.args.get("code")
    print(f"✅ Authorization Code: {code}")
    return "✅ Instagram Business Login successful. You may close this tab."

@app.route("/deauthorize", methods=["POST"])
def deauthorize():
    print("⚠ App deauthorized.")
    return "OK", 200

@app.route("/delete_data", methods=["POST"])
def delete_data():
    print("🗑 Data deletion requested.")
    return {"status": "user data deleted"}, 200

if __name__ == "__main__":
    app.run(port=8501)
