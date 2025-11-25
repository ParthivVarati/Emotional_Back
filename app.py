from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json

from tom_agent import get_tom_chain
from jerry_agent import get_jerry_chain

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# -------------------------------
# Chat History (Stored in JSON)
# -------------------------------

HISTORY_FILE = "chat_histories.json"

# Load existing histories
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        all_histories = json.load(f)
else:
    all_histories = {"tom": [], "jerry": []}


def get_history(bot_id):
    return all_histories.get(bot_id, [])


def save_history(bot_id, history):
    all_histories[bot_id] = history
    with open(HISTORY_FILE, "w") as f:
        json.dump(all_histories, f, indent=2)


# -------------------------------
# Chat Processing Function
# -------------------------------

def process_message(bot_id, user_message):
    history = get_history(bot_id)

    history.append({"role": "user", "content": user_message})
    save_history(bot_id, history)

    formatted_history = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted_history += f"{role}: {msg['content']}\n"

    if bot_id == "tom":
        chain = get_tom_chain(GEMINI_API_KEY)
    else:
        chain = get_jerry_chain(GEMINI_API_KEY)

    final_input = f"""
Continue the conversation below in your assigned persona.
Do NOT repeat your intro every time. Respond naturally.

{formatted_history}
"""

    result = chain.invoke({"input": final_input})
    reply = result.content if hasattr(result, "content") else str(result)

    history.append({"role": "bot", "content": reply})
    save_history(bot_id, history)

    return reply


# -------------------------------
# API Endpoints
# -------------------------------

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "My Feelings Buddy API is running"})


@app.route("/chat/tom", methods=["POST"])
def chat_tom():
    data = request.get_json()
    msg = data.get("message", "")
    reply = process_message("tom", msg)
    return jsonify({"reply": reply})


@app.route("/chat/jerry", methods=["POST"])
def chat_jerry():
    data = request.get_json()
    msg = data.get("message", "")
    reply = process_message("jerry", msg)
    return jsonify({"reply": reply})


# -------------------------------
# Run Server
# -------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
