# Import the operating system interface module for handling file paths and environment variables
import os
# Used for regular expression matching and string processing
import re
# Used for JSON serialization and deserialization
import json
# For defining asynchronous context managers
from contextlib import asynccontextmanager
# For type hints, defining lists and optional parameters
from typing import List, Tuple
# For creating a web application and handling HTTP exceptions
from fastapi import FastAPI, HTTPException, Depends
# For returning JSON and streaming responses
from fastapi.responses import JSONResponse, StreamingResponse
# For running FastAPI applications
import uvicorn
# Import the logging module for recording runtime information
import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
# Import system module for system-level operations like exiting the program
import sys
import time
# Import UUID module for generating unique identifiers
import uuid
# Import typing tools for type annotations
from typing import Optional
# Import base classes and field tools from Pydantic
from pydantic import BaseModel, Field
# Import custom modules and functions
from ragAgent import (
    ToolConfig,
    create_graph,
    save_graph_visualization,
    get_llm,
    get_tools,
    Config,
    ConnectionPool,
    ConnectionPoolError,
    monitor_connection_pool,
)


# Set LangSmith environment variables for tracing application behavior in real time
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = ""


# Configure basic logging, with level set to DEBUG or INFO
logger = logging.getLogger(__name__)
# Set logger level to DEBUG
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
logger.handlers = []  # Clear default handlers
# Use ConcurrentRotatingFileHandler
handler = ConcurrentRotatingFileHandler(
    # Log file
    Config.LOG_FILE,
    # Maximum log file size set to 5MB, triggering rotation when exceeded
    maxBytes=Config.MAX_BYTES,
    # Keep up to 3 backup log files when rotating
    backupCount=Config.BACKUP_COUNT
)
# Set handler level to DEBUG
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)


# Define message classes for API response data encapsulation
# Define the Message class
class Message(BaseModel):
    role: str
    content: str

# Define the ChatCompletionRequest class
class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    userId: Optional[str] = None
    conversationId: Optional[str] = None

# Define the ChatCompletionResponseChoice class
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

# Define the ChatCompletionResponse class
class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    system_fingerprint: Optional[str] = None


def format_response(response):
    """Format input text by adding paragraph breaks, proper line spacing, and code block markers for better readability.

    Args:
        response: Input text.

    Returns:
        Formatted text with clear paragraph separation.
    """
    # Split text into paragraphs using two or more consecutive newlines
    paragraphs = re.split(r'\n{2,}', response)
    # List to store formatted paragraphs
    formatted_paragraphs = []
    # Process each paragraph
    for para in paragraphs:
        # Check if paragraph contains code block markers
        if '```' in para:
            # Split paragraph into parts, alternating between code blocks and plain text
            parts = para.split('```')
            for i, part in enumerate(parts):
                # Odd indices represent code blocks
                if i % 2 == 1:
                    # Wrap code blocks with ``` markers and strip extra spaces
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            para = ''.join(parts)
        else:
            # Replace ". " with ".\n" to separate sentences clearly
            para = para.replace('. ', '.\n')
        # Add formatted paragraph, trimming leading/trailing whitespace
        formatted_paragraphs.append(para.strip())
    # Join all formatted paragraphs with double line breaks
    return '\n\n'.join(formatted_paragraphs)


# Asynchronous context manager for managing FastAPI app lifecycle: initialization and cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the FastAPI app lifecycle: initialization on startup and cleanup on shutdown.

    Args:
        app (FastAPI): FastAPI application instance.

    Yields:
        None: Initialization occurs before yield, cleanup after yield.

    Raises:
        ConnectionPoolError: Raised when connection pool initialization fails.
        Exception: For other unexpected errors.
    """
    global graph, tool_config
    db_connection_pool = None
    try:
        # Initialize chat model and embedding model
        llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

        # Get tool list based on embedding model
        tools = get_tools()

        # Create tool configuration instance
        tool_config = ToolConfig(tools)

        # Define database connection parameters
        connection_kwargs = {"autocommit": True, "prepare_threshold": 0, "connect_timeout": 5}
        # Create a database connection pool
        db_connection_pool = ConnectionPool(
            conninfo=Config.DB_URI,
            max_size=20,
            min_size=2,
            kwargs=connection_kwargs,
            timeout=10
        )

        # Try to open the connection pool
        try:
            db_connection_pool.open()
            logger.info("Database connection pool initialized")
            logger.debug("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to open connection pool: {e}")
            raise ConnectionPoolError(f"Failed to open database connection pool: {str(e)}")

        # Start the connection pool monitor thread (checks every 60s, daemon mode)
        monitor_thread = monitor_connection_pool(db_connection_pool, interval=60)

        # Attempt to create the graph
        try:
            graph = create_graph(db_connection_pool, llm_chat, llm_embedding, tool_config)
        except ConnectionPoolError as e:
            logger.error(f"Graph creation failed: {e}")
            sys.exit(1)

        # Save the visual representation of the graph
        save_graph_visualization(graph)

    except ConnectionPoolError as e:
        logger.error(f"Connection pool error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

    yield
    # Cleanup resources when shutting down
    if db_connection_pool and not db_connection_pool.closed:
        db_connection_pool.close()
        logger.info("Database connection pool closed")
    logger.info("The service has been shut down")

# Create a FastAPI instance; lifespan handles startup and shutdown hooks
app = FastAPI(lifespan=lifespan)


# Handle non-streaming response: generate and return a complete response
async def handle_non_stream_response(user_input, graph, tool_config, config):
    """
    Handles non-streaming responses, generating the complete response output.

    Args:
        user_input (str): User input text.
        graph: Graph object for message flow.
        tool_config: Tool configuration object containing available tools.
        config (dict): Configuration parameters including thread and user IDs.

    Returns:
        JSONResponse: A JSON response containing the formatted output.
    """
    content = None
    try:
        events = graph.stream({"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0}, config)
        for event in events:
            for value in event.values():
                if "messages" not in value or not isinstance(value["messages"], list):
                    logger.warning("No valid messages in response")
                    continue

                last_message = value["messages"][-1]

                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        if isinstance(tool_call, dict) and "name" in tool_call:
                            logger.info(f"Calling tool: {tool_call['name']}")
                    continue

                if hasattr(last_message, "content"):
                    content = last_message.content
                    if hasattr(last_message, "name") and last_message.name in tool_config.get_tool_names():
                        tool_name = last_message.name
                        logger.info(f"Tool Output [{tool_name}]: {content}")
                    else:
                        logger.info(f"Final Response is: {content}")
                else:
                    logger.info("Message has no content, skipping")
    except ValueError as ve:
        logger.error(f"Value error in response processing: {ve}")
    except Exception as e:
        logger.error(f"Error processing response: {e}")

    formatted_response = str(format_response(content)) if content else "No response generated"
    logger.info(f"Results for Formatting: {formatted_response}")

    try:
        response = ChatCompletionResponse(
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=formatted_response),
                    finish_reason="stop"
                )
            ]
        )
    except Exception as resp_error:
        logger.error(f"Error creating response object: {resp_error}")
        response = ChatCompletionResponse(
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content="Error generating response"),
                    finish_reason="error"
                )
            ]
        )

    logger.info(f"Send response content: \n{response}")
    return JSONResponse(content=response.model_dump())


# Handle streaming responses and return stream data
async def handle_stream_response(user_input, graph, config):
    """
    Handles streaming responses, returning real-time chunks of data.

    Args:
        user_input (str): User input text.
        graph: Graph object for message flow.
        config (dict): Configuration parameters including thread and user IDs.

    Returns:
        StreamingResponse: A streaming response with MIME type text/event-stream.
    """
    async def generate_stream():
        """
        Internal async generator that yields streaming response data.

        Yields:
            str: Data chunk formatted as SSE (Server-Sent Events).

        Raises:
            Exception: If stream generation fails.
        """
        try:
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            stream_data = graph.stream(
                {"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0},
                config,
                stream_mode="messages"
            )
            for message_chunk, metadata in stream_data:
                try:
                    node_name = metadata.get("langgraph_node") if metadata else None
                    if node_name in ["generate", "agent"]:
                        chunk = getattr(message_chunk, 'content', '')
                        logger.info(f"Streaming chunk from {node_name}: {chunk}")
                        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
                except Exception as chunk_error:
                    logger.error(f"Error processing stream chunk: {chunk_error}")
                    continue

            yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        except Exception as stream_error:
            logger.error(f"Stream generation error: {stream_error}")
            yield f"data: {json.dumps({'error': 'Stream processing failed'})}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


# Dependency injection function to retrieve graph and tool_config
async def get_dependencies() -> Tuple[any, any]:
    """
    Dependency injection for retrieving graph and tool_config.

    Returns:
        Tuple: (graph, tool_config)

    Raises:
        HTTPException: If graph or tool_config are not initialized.
    """
    if not graph or not tool_config:
        raise HTTPException(status_code=500, detail="Service not initialized")
    return graph, tool_config


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, dependencies: Tuple[any, any] = Depends(get_dependencies)):
    """Receive and process chat completion requests from the frontend.

    Args:
        request: Request parameters.

    Returns:
        Standard Python dictionary.
    """
    try:
        graph, tool_config = dependencies
        if not request.messages or not request.messages[-1].content:
            logger.error("Invalid request: Empty or invalid messages")
            raise HTTPException(status_code=400, detail="Messages cannot be empty or invalid")
        user_input = request.messages[-1].content
        logger.info(f"The user's input is: {user_input}")

        config = {
            "configurable": {
                "thread_id": f"{getattr(request, 'userId', 'unknown')}@@{getattr(request, 'conversationId', 'default')}",
                "user_id": getattr(request, 'userId', 'unknown')
            }
        }

        if request.stream:
            return await handle_stream_response(user_input, graph, config)
        return await handle_non_stream_response(user_input, graph, tool_config, config)

    except Exception as e:
        logger.error(f"Error handling chat completion:\n\n {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info(f"Start the server on port {Config.PORT}")
    # uvicorn is a lightweight, high-performance ASGI server for running FastAPI applications
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
