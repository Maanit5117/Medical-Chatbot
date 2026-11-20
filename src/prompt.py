from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = (
    "You are a medical information assistant. Answer ONLY from the context below. "
    "If the context does not contain the answer, say you don't know. "
    "Do not diagnose or give personal medical advice; this is general information. "
    "Keep answers concise.\n\nContext:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])
