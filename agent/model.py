from langchain.chat_models import init_chat_model


deepseek_r1_distill_qwen3_8b = init_chat_model(
    "openai:deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    temperature=0,
    api_key="your api key",
    base_url="https://api.siliconflow.cn/v1"
)

qwen3_8b = init_chat_model(
    "openai:Qwen/Qwen3-8B",
    temperature=0,
    api_key="your api key",
    base_url="https://api.siliconflow.cn/v1"
)