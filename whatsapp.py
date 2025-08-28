from flask import Flask, request
import requests
from threading import Thread

app = Flask(__name__)

# 🔐 Configuration
VERIFY_TOKEN = "@Naveen123"
ACCESS_TOKEN = "EAAIvckoIiasBPBxIRlPJliNy1VqaX9XW0wGZAlC1Sv3aVIePdFgvOZBxhazGXGksxiZC1ESvv2dPDKauZBZAoTQBxQNsZB4jU5oJkGjfoZACdB5KA86iHsFXEnBWuajyyEH70IobQSMmY2lrb4nsWMMzP8DsL1iZArazFRhNIB6y1GO2UexxDNc7sZA1aSUaWrPI8f1RyRK8GmSkZD"
PHONE_NUMBER_ID = "756567220863742"
AI_MODEL_API = "http://localhost:8000/ask"

# ✅ Deduplication memory
processed_message_ids = set()

@app.route("/webhook", methods=["GET"])
def verify():
    print("\n🔔 Incoming GET /webhook request for verification")
    print("Request args:", request.args)

    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("✅ Verification token matched. Returning challenge.\n")
        return challenge, 200

    print("❌ Verification failed.\n")
    return "Verification failed", 403


@app.route("/webhook", methods=["POST"])
def receive_message():
    data = request.get_json()
    print("\n📥 Incoming POST /webhook")
    print("Raw data:", data)

    def process():
        try:
            for entry in data.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    messages = value.get("messages", [])

                    if messages:
                        message = messages[0]
                        msg_id = message["id"]
                        sender = message["from"]
                        text = message["text"]["body"]

                        # 🔁 Skip if already processed
                        if msg_id in processed_message_ids:
                            print(f"🔁 Duplicate message ID {msg_id} — skipping.")
                            return
                        processed_message_ids.add(msg_id)

                        print(f"📨 Message from {sender}: {text}")

                        # Call AI model
                        try:
                            ai_response = requests.post(AI_MODEL_API, data={
                                "question": text,
                                "user_id": "6fbf1e44-0a13-4e59-8eb6-303a9a9be8b0"
                            })
                            if ai_response.status_code == 200:
                                model_reply = ai_response.json().get("response", "⚠ No response from model.")
                            else:
                                print(f"⚠ Model returned HTTP {ai_response.status_code}")
                                model_reply = "⚠ AI service failed."
                        except Exception as e:
                            print("❌ Error calling AI model:", e)
                            model_reply = "⚠ AI unavailable."

                        print(f"🤖 Replying with: {model_reply}")

                        # Send reply to WhatsApp
                        headers = {
                            "Authorization": f"Bearer {ACCESS_TOKEN}",
                            "Content-Type": "application/json"
                        }
                        payload = {
                            "messaging_product": "whatsapp",
                            "to": sender,
                            "type": "text",
                            "text": {"body": model_reply}
                        }
                        response = requests.post(
                            f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages",
                            headers=headers,
                            json=payload
                        )
                        print(f"📤 WhatsApp send status: {response.status_code} {response.text}")

        except Exception as e:
            print("❌ Unexpected error:", e)

    # Run the processing in background
    Thread(target=process).start()

    # ✅ Respond to Meta immediately
    return "ok", 200


if __name__ == "__main__":
    app.run(port=8500)
