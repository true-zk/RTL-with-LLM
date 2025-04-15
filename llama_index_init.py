from llama_index.llms.openai_like import OpenAILike

from config import (
    MODEL_NAME,
    BASE_URL,
    API_KEY,
)


llm = OpenAILike(
    model=MODEL_NAME,
    api_base=BASE_URL,
    api_key=API_KEY,
    temperature=0.1,
    is_chat_model=True,
)

