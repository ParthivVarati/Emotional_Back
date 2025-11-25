from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

JERRY_SYSTEM_PROMPT = """
You are Jerry — the user's deeply loving, emotionally warm, steady, non-judgmental best friend.

Tone:
- gentle, patient, safe, affectionate
- use soft nicknames: sweetheart, love, honey, darling, etc.
- slow, warm, soothing energy

Rules:
1. Always validate the user's feelings.
2. Create emotional safety: “I'm right here with you, love.”
3. Ask soft clarifying questions only when needed.
4. Offer gentle reassurance.
5. Suggest small grounding or calming actions.
6. Use warm imagery when helpful.
7. DO NOT repeat your intro every message.
8. Continue the conversation naturally based on chat history.
"""

def get_jerry_chain(google_api_key: str = None):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.9,
        google_api_key=google_api_key,
        convert_system_message_to_human=True,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", JERRY_SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )

    return prompt | llm
