import os
import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# Set logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Model configuration dictionary
MODEL_CONFIGS = {
    "openai": {
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "chat_model": "gpt-4o",
        "embedding_model": "text-embedding-3-small"
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "chat_model": "qwen-max",
        "embedding_model": "text-embedding-v1"
    },
    "oneapi": {
        "base_url": "http://139.224.72.218:3000/v1",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "chat_model": "qwen-max",
        "embedding_model": "text-embedding-v1"
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "chat_model": "qwen2.5:32b",
        "embedding_model": "bge-m3:latest"
    }
}


# Default configuration
DEFAULT_LLM_TYPE = "qwen"
DEFAULT_TEMPERATURE = 0.0


class LLMInitializationError(Exception):
    """Custom exception class for LLM initialization errors"""
    pass


def initialize_llm(llm_type: str = DEFAULT_LLM_TYPE) -> tuple[ChatOpenAI, OpenAIEmbeddings]:
    """
    Initialize an LLM instance

    Args:
        llm_type (str): LLM type, one of 'openai', 'oneapi', 'qwen', 'ollama'

    Returns:
        ChatOpenAI: Initialized LLM instance

    Raises:
        LLMInitializationError: Raised when LLM initialization fails
    """
    try:
        # Check if llm_type is valid
        if llm_type not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported LLM type: {llm_type}. Available types: {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[llm_type]

        # Special handling for 'ollama' type
        if llm_type == "ollama":
            os.environ["OPENAI_API_KEY"] = "NA"

        # Create LLM instance
        llm_chat = ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["chat_model"],
            temperature=DEFAULT_TEMPERATURE,
            timeout=30,  # Add timeout setting (seconds)
            max_retries=2  # Add retry attempts
        )

        llm_embedding = OpenAIEmbeddings(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["embedding_model"],
            deployment=config["embedding_model"],
            check_embedding_ctx_length=False
        )

        logger.info(f"Successfully initialized {llm_type} LLM")
        return llm_chat, llm_embedding

    except ValueError as ve:
        logger.error(f"LLM configuration error: {str(ve)}")
        raise LLMInitializationError(f"LLM configuration error: {str(ve)}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        raise LLMInitializationError(f"Failed to initialize LLM: {str(e)}")


def get_llm(llm_type: str = DEFAULT_LLM_TYPE) -> ChatOpenAI:
    """
    Wrapper function to get an LLM instance, with default and error handling

    Args:
        llm_type (str): LLM type

    Returns:
        ChatOpenAI: LLM instance
    """
    try:
        return initialize_llm(llm_type)
    except LLMInitializationError as e:
        logger.warning(f"Retrying with default configuration: {str(e)}")
        if llm_type != DEFAULT_LLM_TYPE:
            return initialize_llm(DEFAULT_LLM_TYPE)
        raise  # Raise exception if default configuration also fails


# Example usage
if __name__ == "__main__":
    try:
        # Test initializing different types of LLMs
        # llm_openai = get_llm("openai")
        llm_qwen = get_llm("qwen")

        # Test invalid type
        # llm_invalid = get_llm("invalid_type")
    except LLMInitializationError as e:
        logger.error(f"Program terminated: {str(e)}")
