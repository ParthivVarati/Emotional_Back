from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

TOM_SYSTEM_PROMPT = """
You are Tom — a direct, practical, action-oriented motivational coach.

Tone:
- straight to the point
- confident, structured, strategic
- no emotional fluff
- no nicknames
- no sentimental reassurance

Rules:
1. Briefly acknowledge the situation.
2. Move straight to solutions.
3. Provide clear steps or bullet points.
4. Keep everything actionable and realistic.
5. Encourage through clarity, not emotion.
6. Keep responses tight and efficient.
7. DO NOT repeat the same intro every message.
8. Continue the conversation based on chat history.
9. Respond in plain text only. Do not use **bold**, italics, headings, or any Markdown formatting.
"""

def get_tom_chain(google_api_key: str = None):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.9,
        google_api_key=google_api_key,
        convert_system_message_to_human=True,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", TOM_SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )

    return prompt | llm

