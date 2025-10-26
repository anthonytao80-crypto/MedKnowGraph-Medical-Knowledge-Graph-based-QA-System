# Import the logging module for recording runtime information
import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
# Import os module for file path and environment variable operations
import os
# Import sys module for system-level operations such as exiting
import sys
import threading
import time
# Import UUID module for generating unique identifiers
import uuid
# Import escape from html to escape HTML special characters
from html import escape
# Import typing utilities for type hints
from typing import Literal, Annotated, Sequence, Optional
# Import TypedDict from typing_extensions to define typed dictionaries
from typing_extensions import TypedDict
# Import LangChain prompt template classes
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# Import LangChain base message class
from langchain_core.messages import BaseMessage
# Import add_messages for message sequence appending
from langgraph.graph.message import add_messages
# Import prebuilt tool condition and tool node
from langgraph.prebuilt import tools_condition, ToolNode
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import ToolMessage
# Import state graph and START/END node definitions
from langgraph.graph import StateGraph, START, END
# Import base storage interface
from langgraph.store.base import BaseStore
# Import runnable configuration class
from langchain_core.runnables import RunnableConfig
# Import Postgres store class
from langgraph.store.postgres import PostgresStore
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# Import OperationalError from psycopg2 for handling database connection errors
from psycopg2 import OperationalError
# Import Postgres checkpoint saver
from langgraph.checkpoint.postgres import PostgresSaver
# Import PostgreSQL connection pool
from psycopg_pool import ConnectionPool
# Import Pydantic BaseModel and Field
from pydantic import BaseModel, Field
# Import custom get_llm function to load the LLM model
from utils.llms import get_llm
# Import tool configuration utilities
from utils.tools_config import get_tools
# Import unified Config class
from utils.config import Config

# # Set up logging configuration, level set to DEBUG or INFO
logger = logging.getLogger(__name__)
# Set logger level to DEBUG
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
logger.handlers = []  # Clear default handlers
# Use ConcurrentRotatingFileHandler
handler = ConcurrentRotatingFileHandler(
    # Log file path
    Config.LOG_FILE,
    # Maximum log file size: 5MB before rotation
    maxBytes = Config.MAX_BYTES,
    # Keep up to 3 backup log files
    backupCount = Config.BACKUP_COUNT
)
# Set handler level to DEBUG
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)


# Define message state class using TypedDict for type annotations
class MessagesState(TypedDict):
    # messages field: a sequence of messages, appended with add_messages
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # relevance_score field: stores document relevance evaluation ('yes' or 'no')
    relevance_score: Annotated[Optional[str], "Relevance score of retrieved documents, 'yes' or 'no'"]
    # rewrite_count field: tracks query rewrite count to stop recursive graph loops
    rewrite_count: Annotated[int, "Number of times query has been rewritten"]

# Define ToolConfig class for managing tool routing configuration
class ToolConfig:
    # Initialization method that receives a list of tools
    def __init__(self, tools):
        # Store provided tools list
        self.tools = tools
        # Create a set of tool names
        self.tool_names = {tool.name for tool in tools}
        # Build dynamic routing configuration
        self.tool_routing_config = self._build_routing_config(tools)
        # Log initialization result for debugging
        logger.info(f"Initialized ToolConfig with tools: {self.tool_names}, routing: {self.tool_routing_config}")

    # Internal method to dynamically build tool routing configuration
    def _build_routing_config(self, tools):
        # Create empty dict for name-to-node mapping
        routing_config = {}
        # Iterate through each tool
        for tool in tools:
            # Convert tool name to lowercase for case-insensitive matching
            tool_name = tool.name.lower()
            # Route retrieval tools to 'grade_documents'
            if "retrieve" in tool_name:
                routing_config[tool_name] = "grade_documents"
                logger.debug(f"Tool '{tool_name}' routed to 'grade_documents' (retrieval tool)")
            # Non-retrieval tools route to 'generate'
            else:
                routing_config[tool_name] = "generate"
                logger.debug(f"Tool '{tool_name}' routed to 'generate' (non-retrieval tool)")
        # If routing config is empty, log warning
        if not routing_config:
            logger.warning("No tools provided or routing config is empty")
        return routing_config

    # Return the tool list
    def get_tools(self):
        return self.tools

    # Return the tool name set
    def get_tool_names(self):
        return self.tool_names

    # Return the routing config
    def get_tool_routing_config(self):
        return self.tool_routing_config

# Document relevance scoring model
class DocumentRelevanceScore(BaseModel):
    # binary_score: 'yes' or 'no' relevance label
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

# Custom exception for database connection pool errors
class ConnectionPoolError(Exception):
    """Custom exception indicating connection pool initialization or status errors"""
    pass

# Redefine ToolNode to support concurrent tool calls
class ParallelToolNode(ToolNode):
    # Initialize with tools and optional thread pool size
    def __init__(self, tools, max_workers: int = 5):
        super().__init__(tools)
        self.max_workers = max_workers  # Maximum thread count

    # Private method to execute a single tool call
    def _run_single_tool(self, tool_call: dict, tool_map: dict) -> ToolMessage:
        """Execute a single tool call"""
        try:
            tool_name = tool_call["name"]
            tool = tool_map.get(tool_name)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")
            result = tool.invoke(tool_call["args"])
            return ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool_name
            )
        except Exception as e:
            logger.error(f"Error executing tool {tool_call.get('name', 'unknown')}: {e}")
            return ToolMessage(
                content=f"Error: {str(e)}",
                tool_call_id=tool_call["id"],
                name=tool_call.get("name", "unknown")
            )

    # Callable entry: execute all tool calls in parallel
    def __call__(self, state: dict) -> dict:
        """Execute all tool calls in parallel"""
        logger.info("ParallelToolNode processing tool calls")
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        if not tool_calls:
            logger.warning("No tool calls found in state")
            return {"messages": []}

        tool_map = {tool.name: tool for tool in self.tools}
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_tool = {
                executor.submit(self._run_single_tool, tool_call, tool_map): tool_call
                for tool_call in tool_calls
            }
            for future in as_completed(future_to_tool):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Tool execution failed: {e}")
                    tool_call = future_to_tool[future]
                    results.append(ToolMessage(
                        content=f"Unexpected error: {str(e)}",
                        tool_call_id=tool_call["id"],
                        name=tool_call.get("name", "unknown")
                    ))

        logger.info(f"Completed {len(results)} tool calls")
        return {"messages": results}


# Helper function to get the latest user question
def get_latest_question(state: MessagesState) -> Optional[str]:
    """Safely get the latest user question from state"""
    try:
        if not state.get("messages") or not isinstance(state["messages"], (list, tuple)) or len(state["messages"]) == 0:
            logger.warning("No valid messages found in state for getting latest question")
            return None

        for message in reversed(state["messages"]):
            if message.__class__.__name__ == "HumanMessage" and hasattr(message, "content"):
                return message.content

        logger.info("No HumanMessage found in state")
        return None

    except Exception as e:
        logger.error(f"Error getting latest question: {e}")
        return None


# Thread-level message filtering for persistence
def filter_messages(messages: list) -> list:
    """Filter message list, keeping only AIMessage and HumanMessage types"""
    filtered = [msg for msg in messages if msg.__class__.__name__ in ['AIMessage', 'HumanMessage']]
    return filtered[-5:] if len(filtered) > 5 else filtered


# Cross-thread storage and filtering for persistent memory
def store_memory(question: BaseMessage, config: RunnableConfig, store: BaseStore) -> str:
    """Store user-related memory information."""
    namespace = ("memories", config["configurable"]["user_id"])
    try:
        memories = store.search(namespace, query=str(question.content))
        user_info = "\n".join([d.value["data"] for d in memories])

        if "记住" in question.content.lower():
            memory = escape(question.content)
            store.put(namespace, str(uuid.uuid4()), {"data": memory})
            logger.info(f"Stored memory: {memory}")

        return user_info
    except Exception as e:
        logger.error(f"Error in store_memory: {e}")
        return ""


# Create a processing chain for LLM
def create_chain(llm_chat, template_file: str, structured_output=None):
    """Create an LLM processing chain with caching to avoid reloading templates."""
    if not hasattr(create_chain, "prompt_cache"):
        create_chain.prompt_cache = {}
        create_chain.lock = threading.Lock()

    try:
        if template_file in create_chain.prompt_cache:
            prompt_template = create_chain.prompt_cache[template_file]
            logger.info(f"Using cached prompt template for {template_file}")
        else:
            with create_chain.lock:
                if template_file not in create_chain.prompt_cache:
                    logger.info(f"Loading and caching prompt template from {template_file}")
                    create_chain.prompt_cache[template_file] = PromptTemplate.from_file(template_file, encoding="utf-8")
                prompt_template = create_chain.prompt_cache[template_file]

        prompt = ChatPromptTemplate.from_messages([("human", prompt_template.template)])
        return prompt | (llm_chat.with_structured_output(structured_output) if structured_output else llm_chat)
    except FileNotFoundError:
        logger.error(f"Template file {template_file} not found")
        raise


# Database retry mechanism — up to 3 retries, exponential backoff between 2–10 seconds
@retry(stop=stop_after_attempt(3),wait=wait_exponential(multiplier=1, min=2, max=10),retry=retry_if_exception_type(OperationalError))
def test_connection(db_connection_pool: ConnectionPool) -> bool:
    """Test whether connection pool is available"""
    with db_connection_pool.getconn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            if result != (1,):
                raise ConnectionPoolError("Connection pool test query failed, unexpected result")
    return True


# Periodically check pool status and log connection usage or anomalies
def monitor_connection_pool(db_connection_pool: ConnectionPool, interval: int = 60):
    """Monitor connection pool status periodically"""
    def _monitor():
        while not db_connection_pool.closed:
            try:
                stats = db_connection_pool.get_stats()
                active = stats.get("connections_in_use", 0)
                total = db_connection_pool.max_size
                logger.info(f"Connection db_connection_pool status: {active}/{total} connections in use")
                if active >= total * 0.8:
                    logger.warning(f"Connection db_connection_pool nearing capacity: {active}/{total}")
            except Exception as e:
                logger.error(f"Failed to monitor connection db_connection_pool: {e}")
            time.sleep(interval)

    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()
    return monitor_thread


# Define Node: agent triage function
def agent(state: MessagesState, config: RunnableConfig, *, store: BaseStore, llm_chat, tool_config: ToolConfig) -> dict:
    """Agent function to decide whether to call a tool or finish."""
    logger.info("Agent processing user query")
    namespace = ("memories", config["configurable"]["user_id"])
    try:
        question = state["messages"][-1]
        logger.info(f"agent question:{question}")

        user_info = store_memory(question, config, store)
        messages = filter_messages(state["messages"])

        llm_chat_with_tool = llm_chat.bind_tools(tool_config.get_tools())

        agent_chain = create_chain(llm_chat_with_tool, Config.PROMPT_TEMPLATE_TXT_AGENT)
        response = agent_chain.invoke({"question": question,"messages": messages, "userInfo": user_info})
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in agent processing: {e}")
        return {"messages": [{"role": "system", "content": "Error processing request"}]}


# Define Node: grade_documents relevance evaluation function
def grade_documents(state: MessagesState, llm_chat) -> dict:
    """Evaluate relevance between retrieved documents and user query."""
    logger.info("Grading documents for relevance")
    if not state.get("messages"):
        logger.error("Messages state is empty")
        return {
            "messages": [{"role": "system", "content": "State empty, cannot grade"}],
            "relevance_score": None
        }

    try:
        question = get_latest_question(state)
        context = state["messages"][-1].content

        grade_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_GRADE, DocumentRelevanceScore)
        scored_result = grade_chain.invoke({"question": question, "context": context})
        score = scored_result.binary_score
        logger.info(f"Document relevance score: {score}")

        return {
            "messages": state["messages"],
            "relevance_score": score
        }
    except (IndexError, KeyError) as e:
        logger.error(f"Message access error: {e}")
        return {
            "messages": [{"role": "system", "content": "Cannot grade document"}],
            "relevance_score": None
        }
    except Exception as e:
        logger.error(f"Unexpected error in grading: {e}")
        return {
            "messages": [{"role": "system", "content": "Error during grading"}],
            "relevance_score": None
        }


# Query rewriting
def rewrite(state: MessagesState, llm_chat) -> dict:
    """Rewrite user query to improve clarity."""
    logger.info("Rewriting query")
    try:
        question = get_latest_question(state)
        rewrite_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_REWRITE)
        response = rewrite_chain.invoke({"question": question})
        rewrite_count = state.get("rewrite_count", 0) + 1
        logger.info(f"Rewrite count: {rewrite_count}")
        return {"messages": [response], "rewrite_count": rewrite_count}
    except (IndexError, KeyError) as e:
        logger.error(f"Message access error in rewrite: {e}")
        return {"messages": [{"role": "system", "content": "Cannot rewrite query"}]}



# Define Node: Generate response function
def generate(state: MessagesState, llm_chat) -> dict:
    """Generate the final response based on the tool output.

    Args:
        state: Current conversation state.

    Returns:
        dict: Updated message state.
    """
    # Log start of response generation
    logger.info("Generating final response")
    # Try to execute the following block
    try:
        # Get the user's latest question
        question = get_latest_question(state)
        # Get the last message as context (since tool outputs are written to the latest message in state)
        context = state["messages"][-1].content
        # logger.info(f"generate - Question: {question}, Context: {context}")
        # Create the generation chain
        generate_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_GENERATE)
        # Call the generation chain to produce a response
        response = generate_chain.invoke({"context": context, "question": question})
        # Return the updated message state
        return {"messages": [response]}
    # Catch index or key errors
    except (IndexError, KeyError) as e:
        # Log the error
        logger.error(f"Message access error in generate: {e}")
        # Return an error message
        return {"messages": [{"role": "system", "content": "Unable to generate a response"}]}


# Define Edge: Dynamically decide next route based on tool results
def route_after_tools(state: MessagesState, tool_config: ToolConfig) -> Literal["generate", "grade_documents"]:
    """
    Dynamically determine the next route based on the result of tool invocation,
    supporting multiple tools with a configuration dictionary and fault tolerance.

    Args:
        state: Current conversation state, containing message history and possible tool results.
        tool_config: Tool configuration parameters.

    Returns:
        Literal["generate", "grade_documents"]: Next target node.
    """
    # Check if state contains messages; if empty, log error and default to generate
    if not state.get("messages") or not isinstance(state["messages"], list):
        logger.error("Messages state is empty or invalid, defaulting to generate")
        return "generate"

    try:
        # Get the last message to determine the source of tool invocation
        last_message = state["messages"][-1]

        # If the message has no name attribute, route to generate
        if not hasattr(last_message, "name") or last_message.name is None:
            logger.info("Last message has no name attribute, routing to generate")
            return "generate"

        # Check if the message comes from a registered tool
        tool_name = last_message.name
        if tool_name not in tool_config.get_tool_names():
            logger.info(f"Unknown tool {tool_name}, routing to generate")
            return "generate"

        # Determine route based on config dictionary, default to generate if missing
        target = tool_config.get_tool_routing_config().get(tool_name, "generate")
        logger.info(f"Tool {tool_name} routed to {target} based on config")
        return target

    except IndexError:
        # Catch list index errors and default to generate
        logger.error("No messages available in state, defaulting to generate")
        return "generate"
    except AttributeError:
        # Catch attribute access errors and default to generate
        logger.error("Invalid message object, defaulting to generate")
        return "generate"
    except Exception as e:
        # Catch unexpected exceptions, log details, and default to generate
        logger.error(f"Unexpected error in route_after_tools: {e}, defaulting to generate")
        return "generate"


# Define Edge: Decide next route based on grading results
def route_after_grade(state: MessagesState) -> Literal["generate", "rewrite"]:
    """
    Determine the next route based on the grading result in the state,
    with enhanced validation and fault tolerance.

    Args:
        state: Current conversation state, expected to contain messages and relevance_score.

    Returns:
        Literal["generate", "rewrite"]: Next target node.
    """
    # Validate that state is a dictionary; default to rewrite if invalid
    if not isinstance(state, dict):
        logger.error("State is not a valid dictionary, defaulting to rewrite")
        return "rewrite"

    # Check if messages field exists and is valid; default to rewrite otherwise
    if "messages" not in state or not isinstance(state["messages"], (list, tuple)):
        logger.error("State missing valid messages field, defaulting to rewrite")
        return "rewrite"

    # If messages list is empty, log warning and default to rewrite
    if not state["messages"]:
        logger.warning("Messages list is empty, defaulting to rewrite")
        return "rewrite"

    # Get relevance_score (if any)
    relevance_score = state.get("relevance_score")
    # Get rewrite count
    rewrite_count = state.get("rewrite_count", 0)
    logger.info(f"Routing based on relevance_score: {relevance_score}, rewrite_count: {rewrite_count}")

    # Force route to generate if rewrite count exceeds 3
    if rewrite_count >= 3:
        tool_result = state["messages"][-1].content
        if not list(tool_result):
            state["messages"][-1].content = "The question and agent task are unrelated"
        logger.info("Max rewrite limit reached, proceeding to generate")
        return "generate"

    try:
        # If relevance_score is not a string, consider it invalid
        if not isinstance(relevance_score, str):
            logger.warning(f"Invalid relevance_score type: {type(relevance_score)}, defaulting to rewrite")
            return "rewrite"

        # If score is "yes", the document is relevant → route to generate
        if relevance_score.lower() == "yes":
            logger.info("Documents are relevant, proceeding to generate")
            return "generate"

        # If score is "no" or other value, route to rewrite
        logger.info("Documents are not relevant or scoring failed, proceeding to rewrite")
        return "rewrite"

    except AttributeError:
        # Catch cases where relevance_score does not support lower()
        logger.error("relevance_score is not a string or is None, defaulting to rewrite")
        return "rewrite"
    except Exception as e:
        # Catch unexpected exceptions
        logger.error(f"Unexpected error in route_after_grade: {e}, defaulting to rewrite")
        return "rewrite"


# Save visualized state graph
def save_graph_visualization(graph: StateGraph, filename: str = "graph.png") -> None:
    """Save a visualization of the state graph.

    Args:
        graph: State graph instance.
        filename: File path to save.
    """
    try:
        # Open file in binary write mode
        with open(filename, "wb") as f:
            # Convert the graph to Mermaid PNG and write to file
            f.write(graph.get_graph().draw_mermaid_png())
        # Log success
        logger.info(f"Graph visualization saved as {filename}")
    except IOError as e:
        # Log warning if failed
        logger.warning(f"Failed to save graph visualization: {e}")


# Create and configure the state graph
def create_graph(db_connection_pool: ConnectionPool, llm_chat, llm_embedding, tool_config: ToolConfig) -> StateGraph:
    """Create and configure the state graph.

    Args:
        db_connection_pool: Database connection pool.
        llm_chat: Chat model.
        llm_embedding: Embedding model.
        tool_config: Tool configuration.

    Returns:
        StateGraph: Compiled state graph.

    Raises:
        ConnectionPoolError: If the pool is uninitialized or invalid.
    """
    # Check if connection pool is None or closed
    if db_connection_pool is None or db_connection_pool.closed:
        logger.error("Connection db_connection_pool is None or closed")
        raise ConnectionPoolError("Database connection pool not initialized or closed")
    try:
        # Get current active and max connections
        active_connections = db_connection_pool.get_stats().get("connections_in_use", 0)
        max_connections = db_connection_pool.max_size
        if active_connections >= max_connections:
            logger.error(f"Connection pool exhausted: {active_connections}/{max_connections} in use")
            raise ConnectionPoolError("Connection pool exhausted, no available connections")
        if not test_connection(db_connection_pool):
            raise ConnectionPoolError("Connection pool test failed")
        logger.info("Connection db_connection_pool status: OK, test successful")
    except OperationalError as e:
        logger.error(f"Database operational error during connection test: {e}")
        raise ConnectionPoolError(f"Connection pool test failed, possibly closed or timed out: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to verify connection db_connection_pool status: {e}")
        raise ConnectionPoolError(f"Failed to verify connection pool status: {str(e)}")

    # Thread-local persistence
    try:
        # Create PostgresSaver checkpoint instance
        checkpointer = PostgresSaver(db_connection_pool)
        # Initialize checkpoint
        checkpointer.setup()
    except Exception as e:
        logger.error(f"Failed to setup PostgresSaver: {e}")
        raise ConnectionPoolError(f"Checkpoint initialization failed: {str(e)}")

    # Cross-thread persistence
    try:
        # Create PostgresStore instance with embedding dimension and function
        store = PostgresStore(db_connection_pool, index={"dims": 1536, "embed": llm_embedding})
        store.setup()
    except Exception as e:
        logger.error(f"Failed to setup PostgresStore: {e}")
        raise ConnectionPoolError(f"Store initialization failed: {str(e)}")

    # Create StateGraph using MessagesState as the state type
    workflow = StateGraph(MessagesState)
    # Add agent node
    workflow.add_node("agent", lambda state, config: agent(state, config, store=store, llm_chat=llm_chat, tool_config=tool_config))
    # Add parallel tool node
    workflow.add_node("call_tools", ParallelToolNode(tool_config.get_tools(), max_workers=5))
    # Add rewrite node
    workflow.add_node("rewrite", lambda state: rewrite(state,llm_chat=llm_chat))
    # Add generate node
    workflow.add_node("generate", lambda state: generate(state, llm_chat=llm_chat))
    # Add document relevance grading node
    workflow.add_node("grade_documents", lambda state: grade_documents(state, llm_chat=llm_chat))

    # Add edges between nodes
    workflow.add_edge(START, end_key="agent")
    workflow.add_conditional_edges(source="agent", path=tools_condition, path_map={"tools": "call_tools", END: END})
    workflow.add_conditional_edges(source="call_tools", path=lambda state: route_after_tools(state, tool_config), path_map={"generate": "generate", "grade_documents": "grade_documents"})
    workflow.add_conditional_edges(source="grade_documents", path=route_after_grade, path_map={"generate": "generate", "rewrite": "rewrite"})
    workflow.add_edge(start_key="generate", end_key=END)
    workflow.add_edge(start_key="rewrite", end_key="agent")

    # Compile state graph and bind checkpoint and store
    return workflow.compile(checkpointer=checkpointer, store=store)


# Define response function
def graph_response(graph: StateGraph, user_input: str, config: dict, tool_config: ToolConfig) -> None:
    """Process user input and output responses, distinguishing between tool and model outputs, supporting multiple tools.

    Args:
        graph: State graph instance.
        user_input: User input.
        config: Runtime configuration.
    """
    try:
        # Start state graph stream
        events = graph.stream({"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0}, config)
        # Iterate over event stream
        for event in events:
            for value in event.values():
                # Validate message existence
                if "messages" not in value or not isinstance(value["messages"], list):
                    logger.warning("No valid messages in response")
                    continue

                # Get last message
                last_message = value["messages"][-1]

                # If message contains tool calls
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        if isinstance(tool_call, dict) and "name" in tool_call:
                            logger.info(f"Calling tool: {tool_call['name']}")
                    continue

                # If message has content
                if hasattr(last_message, "content"):
                    content = last_message.content

                    # Case 1: Tool output
                    if hasattr(last_message, "name") and last_message.name in tool_config.get_tool_names():
                        tool_name = last_message.name
                        print(f"Tool Output [{tool_name}]: {content}")
                    # Case 2: Model output
                    else:
                        print(f"Assistant: {content}")
                else:
                    logger.info("Message has no content, skipping")
                    print("Assistant: No related response retrieved")
    except ValueError as ve:
        logger.error(f"Value error in response processing: {ve}")
        print("Assistant: Value error occurred while processing response")
    except Exception as e:
        logger.error(f"Error processing response: {e}")
        print("Assistant: Unknown error occurred while processing response")


# Define main function
def main():
    """Main function: initialize and run chatbot."""
    db_connection_pool = None
    try:
        # Initialize Chat and Embedding model instances
        llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

        # Get tools list
        tools = get_tools()

        # Create ToolConfig instance
        tool_config = ToolConfig(tools)

        # Define DB connection params: autocommit, no prepared threshold, 5s timeout
        connection_kwargs = {"autocommit": True, "prepare_threshold": 0, "connect_timeout": 5}
        # Create connection pool (max 20, min 2, 10s timeout)
        db_connection_pool = ConnectionPool(conninfo=Config.DB_URI, max_size=20, min_size=2, kwargs=connection_kwargs, timeout=10)

        # Open connection pool
        try:
            db_connection_pool.open()
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to open connection pool: {e}")
            raise ConnectionPoolError(f"Unable to open database connection pool: {str(e)}")

        # Start pool monitoring thread (daemon)
        monitor_thread = monitor_connection_pool(db_connection_pool, interval=60)

        # Create state graph
        try:
            graph = create_graph(db_connection_pool, llm_chat, llm_embedding, tool_config)
        except ConnectionPoolError as e:
            logger.error(f"Graph creation failed: {e}")
            print(f"Error: {e}")
            sys.exit(1)

        # Save graph visualization (optional)
        # save_graph_visualization(graph)

        print("Chatbot ready! Type 'quit', 'exit', or 'q' to end conversation.")
        config = {"configurable": {"thread_id": "1", "user_id": "1"}}

        # Main loop
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break
            if not user_input:
                print("Please enter something to chat!")
                continue
            graph_response(graph, user_input, config, tool_config)

    except ConnectionPoolError as e:
        logger.error(f"Connection pool error: {e}")
        print(f"Error: Database connection pool issue - {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Initialization error: {e}")
        print(f"Error: Initialization failed - {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Goodbye!")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Error: Unknown error occurred - {e}")
        sys.exit(1)
    finally:
        if db_connection_pool and not db_connection_pool.closed:
            db_connection_pool.close()
            logger.info("Database connection pool closed")


if __name__ == "__main__":
    main()
