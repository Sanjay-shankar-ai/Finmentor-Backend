from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.utils import executor
import aiohttp
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_URL = "http://localhost:8000/ask"  # Update if hosted externally

# Initialize bot and dispatcher
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=["start"])
async def start_command(message: Message):
    welcome_text = (
        "👋 Welcome to Fin Mentor – Your AI Financial Assistant!\n\n"
        "Ask me anything about:\n"
        "• 💰 Balance & expenses\n"
        "• 📊 Spending analytics\n"
        "• 🧾 Tax estimation\n"
        "• 🔔 Loan reminders\n"
        "• 🏦 Loan optimization\n\n"
        "Type your question below ⤵️"
    )
    await message.answer(welcome_text)

@dp.message_handler()
async def handle_user_query(message: Message):
    user_input = message.text
    user_id = "6fbf1e44-0a13-4e59-8eb6-303a9a9be8b0"  # Demo user

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                API_URL,
                data={
                    "question": user_input,
                    "user_id": user_id,
                    "plain_text_mode": "true"
                }
            ) as response:
                print("Status:", response.status)
                raw = await response.text()
                print("Raw response:", raw)

                try:
                    result = await response.json()
                    reply = result.get("response", "❌ Unexpected response format.")
                except Exception as json_err:
                    reply = f"⚠️ JSON decode error: {json_err}"

                await message.answer(reply)

        except Exception as e:
            await message.answer("⚠️ Error: Unable to reach the assistant. Please try again later.")
            print("Telegram Bot Error:", e)

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
