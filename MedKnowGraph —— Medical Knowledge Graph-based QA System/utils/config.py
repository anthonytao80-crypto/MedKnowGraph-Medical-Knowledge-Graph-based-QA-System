# config.py
import os

class Config:
    """Unified configuration class to centrally manage all constants"""

    # Path to prompt template files
    PROMPT_TEMPLATE_TXT_AGENT = "prompts/prompt_template_agent.txt"
    PROMPT_TEMPLATE_TXT_GRADE = "prompts/prompt_template_grade.txt"
    PROMPT_TEMPLATE_TXT_REWRITE = "prompts/prompt_template_rewrite.txt"
    PROMPT_TEMPLATE_TXT_GENERATE = "prompts/prompt_template_generate.txt"

    # Chroma database configuration
    CHROMADB_DIRECTORY = "chromaDB"
    CHROMADB_COLLECTION_NAME = "demo001"

    # Log persistence configuration
    LOG_FILE = "output/app.log"
    MAX_BYTES = 5 * 1024 * 1024
    BACKUP_COUNT = 3

    # Database URI with a default value
    DB_URI = os.getenv("DB_URI", "postgresql://anthony:123456@localhost:5432/postgres?sslmode=disable")

    # LLM type selection:
    # 'openai'  → calls OpenAI GPT models
    # 'qwen'    → calls Alibaba Qwen models
    # 'oneapi'  → calls models supported by OneAPI
    # 'ollama'  → calls local open-source LLMs
    LLM_TYPE = "qwen"

    # API service host and port
    HOST = "0.0.0.0"
    PORT = 8012
