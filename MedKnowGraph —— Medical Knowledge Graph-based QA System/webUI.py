# Import Gradio library for building interactive front-end interfaces
import gradio as gr
# Import requests library for sending HTTP requests
import requests
# Import json library for handling JSON data
import json
# Import logging library for logging
import logging
# Import re library for regular expression operations
import re
# Import uuid library for generating unique identifiers
import uuid
# Import datetime library for handling dates and times
from datetime import datetime

# Set basic logging configuration, specify log level as INFO, and define log format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Create a logger for the current module
logger = logging.getLogger(__name__)

# Define the URL of the backend service API
url = "http://localhost:8012/v1/chat/completions"
# Define HTTP request headers, specifying content type as JSON
headers = {"Content-Type": "application/json"}

# Flag to enable or disable streaming output
stream_flag = True # False

# Initialize an empty dictionary to simulate a user database
users_db = {}
# Initialize an empty dictionary to store mappings between usernames and user IDs
user_id_map = {}

# Define a function to generate a unique user ID
def generate_unique_user_id(username):
    # If the username is not already in the mapping, generate a new UUID
    if username not in user_id_map:
        user_id = str(uuid.uuid4())
        # Ensure the generated ID is not already used, regenerate if necessary
        while user_id in user_id_map.values():
            user_id = str(uuid.uuid4())
        # Store the mapping between username and generated ID
        user_id_map[username] = user_id
    # Return the unique ID for this user
    return user_id_map[username]

# Define a function to generate a unique conversation ID
def generate_unique_conversation_id(username):
    # Return a conversation ID by concatenating the username and a UUID
    return f"{username}_{uuid.uuid4()}"

# Define a function to send messages, process user input, and get backend responses
def send_message(user_message, history, user_id, conversation_id, username):
    # Construct the data to send to the backend, including user message, user ID, and conversation ID
    data = {
        "messages": [{"role": "user", "content": user_message}],
        "stream": stream_flag,
        "userId": user_id,
        "conversationId": conversation_id
    }

    # Update chat history with the user's message and a temporary placeholder response
    history = history + [["user", user_message], ["assistant", "Generating response..."]]
    # First yield: return current chat history and title (title not updated yet)
    yield history, history, None

    # If this is the first message, set the conversation title as the first 20 characters of the user message
    if username and conversation_id:
        if not users_db[username]["conversations"][conversation_id].get("title_set", False):
            new_title = user_message[:20] if len(user_message) > 20 else user_message
            users_db[username]["conversations"][conversation_id]["title"] = new_title
            users_db[username]["conversations"][conversation_id]["title_set"] = True

    # Define a function to format the assistant's response
    def format_response(full_text):
        # Replace <think> tag with bold "Thought process" title
        formatted_text = re.sub(r'<think>', '**Thought process**:\n', full_text)
        # Replace </think> tag with bold "Final response" title
        formatted_text = re.sub(r'</think>', '\n\n**Final response**:\n', full_text)
        # Return the formatted text with whitespace stripped
        return formatted_text.strip()

    # Streamed output
    if stream_flag:
        assistant_response = ""
        try:
            with requests.post(url, headers=headers, data=json.dumps(data), stream=True) as response:
                for line in response.iter_lines():
                    if line:
                        json_str = line.decode('utf-8').strip("data: ")
                        if not json_str:
                            logger.info(f"Received empty string, skipping...")
                            continue
                        if json_str.startswith('{') and json_str.endswith('}'):
                            try:
                                response_data = json.loads(json_str)
                                if 'delta' in response_data['choices'][0]:
                                    content = response_data['choices'][0]['delta'].get('content', '')
                                    # Format the response in real-time
                                    formatted_content = format_response(content)
                                    logger.info(f"Received data: {formatted_content}")
                                    assistant_response += formatted_content
                                    updated_history = history[:-1] + [["assistant", assistant_response]]
                                    yield updated_history, updated_history, None
                                if response_data.get('choices', [{}])[0].get('finish_reason') == "stop":
                                    logger.info(f"Finished receiving JSON data")
                                    break
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON decode error: {e}")
                                yield history[:-1] + [["assistant", "Error parsing response, please try again later."]]
                                break
                        else:
                            logger.info(f"Invalid JSON format: {json_str}")
                    else:
                        logger.info(f"Received empty line")
                else:
                    logger.info("Stream response ended but not explicitly finished")
                    yield history[:-1] + [["assistant", "Did not receive complete response."]]
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            yield history[:-1] + [["assistant", "Request failed, please try again later."]]

    # Non-streamed output
    else:
        # Send POST request to backend and get the response
        response = requests.post(url, headers=headers, data=json.dumps(data))
        # Parse the response as JSON
        response_json = response.json()
        # Extract assistant's reply
        assistant_content = response_json['choices'][0]['message']['content']
        # Format the assistant's reply
        formatted_content = format_response(assistant_content)
        # Update chat history by replacing placeholder with formatted response
        updated_history = history[:-1] + [["assistant", formatted_content]]
        # Second yield: return updated chat history and title (title still not updated)
        yield updated_history, updated_history, None

# Define a function to register a new user
def register(username, password):
    # If username already exists, return an error
    if username in users_db:
        return "Username already exists!"
    # Generate a unique user ID
    user_id = generate_unique_user_id(username)
    # Add new user to the database
    users_db[username] = {"password": password, "user_id": user_id, "conversations": {}}
    # Return success message
    return "Registration successful! Please close the popup and log in."

# Define a function for user login
def login(username, password):
    # Check if username exists and password matches
    if username in users_db and users_db[username]["password"] == password:
        # Get user ID
        user_id = users_db[username]["user_id"]
        # Generate a new conversation ID
        conversation_id = generate_unique_conversation_id(username)
        # Get current time as conversation creation time
        create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Add a new conversation record for the user
        users_db[username]["conversations"][conversation_id] = {
            "history": [],
            "title": "New Chat",
            "create_time": create_time,
            "title_set": False
        }
        # Return login success result and related info
        return True, username, user_id, conversation_id, "Login successful!"
    # If login fails, return error
    return False, None, None, None, "Invalid username or password!"


# Define a function to create a new conversation
def new_conversation(username):
    # If the user is not logged in, return a prompt
    if username not in users_db:
        return "Please log in first!", None
    # Generate a new conversation ID
    conversation_id = generate_unique_conversation_id(username)
    # Get the current time as the conversation creation time
    create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Add a new conversation record to the user database
    users_db[username]["conversations"][conversation_id] = {
        "history": [],
        "title": "New Chat",
        "create_time": create_time,
        "title_set": False
    }
    # Return success message and the new conversation ID
    return "New conversation created successfully!", conversation_id

# Define a function to get the list of conversations
def get_conversation_list(username):
    # If the user is not logged in or has no conversations, return the default option
    if username not in users_db or not users_db[username]["conversations"]:
        return ["Please select a conversation"]
    # Initialize the conversation list
    conv_list = []
    # Iterate over all user conversations
    for conv_id, details in users_db[username]["conversations"].items():
        # Get the conversation title, default to "Untitled Conversation"
        title = details.get("title", "Untitled Conversation")
        # Get the conversation creation time
        create_time = details.get("create_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # Append title and time to the list
        conv_list.append(f"{title} - {create_time}")
    # Return the conversation list including the default option
    return ["Please select a conversation"] + conv_list

# Define a function to extract the conversation ID from the selected option
def extract_conversation_id(selected_option, username):
    # If the default option is selected or user not logged in, return None
    if selected_option == "Please select a conversation" or not username in users_db:
        return None
    # Iterate over all user conversations
    for conv_id, details in users_db[username]["conversations"].items():
        # Get the conversation title and creation time
        title = details.get("title", "Untitled Conversation")
        create_time = details.get("create_time", "")
        # If the option matches, return the corresponding conversation ID
        if f"{title} - {create_time}" == selected_option:
            return conv_id
    # If no match found, return None
    return None

# Define a function to load conversation history
def load_conversation(username, selected_option):
    # If the default option is selected or user not logged in, return empty history
    if selected_option == "Please select a conversation" or not username in users_db:
        return []
    # Extract the conversation ID from the selected option
    conversation_id = extract_conversation_id(selected_option, username)
    # If the conversation ID exists, return the corresponding chat history
    if conversation_id in users_db[username]["conversations"]:
        return users_db[username]["conversations"][conversation_id]["history"]
    # Otherwise, return empty history
    return []

# Use Gradio Blocks to create the front-end interface
# .login-container → layout for login page
# .modal → modal style (registration / conversation history)
# .chat-area → chat area layout
# .header → welcome text at top of chat page
# .header-btn → buttons in the header
with gr.Blocks(title="Chat Assistant", css="""
    .login-container { max-width: 400px; margin: 0 auto; padding-top: 100px; }
    .modal { position: fixed; top: 20%; left: 50%; transform: translateX(-50%); background: white; padding: 20px; max-width: 400px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); border-radius: 8px; z-index: 1000; }
    .chat-area { padding: 20px; height: 80vh; }
    .header { display: flex; justify-content: space-between; align-items: center; padding: 10px; }
    .header-btn { margin-left: 10px; padding: 5px 10px; font-size: 14px; }
""") as demo:
    # Define state variables to track login status
    logged_in = gr.State(False)
    # Define state variable to store current username
    current_user = gr.State(None)
    # Define state variable to store current user ID
    current_user_id = gr.State(None)
    # Define state variable to store current conversation ID
    current_conversation = gr.State(None)
    # Define state variable to store chat history
    chatbot_history = gr.State([])
    # Define state variable to store conversation title
    conversation_title = gr.State("New Chat")

    # Define login page layout, initially visible
    with gr.Column(visible=True, elem_classes="login-container") as login_page:
        # Display title
        gr.Markdown("## Chat Assistant")
        # Username input box
        login_username = gr.Textbox(label="Username", placeholder="Enter your username")
        # Password input box (hidden input)
        login_password = gr.Textbox(label="Password", placeholder="Enter your password", type="password")
        # Create a row layout for login and register buttons
        with gr.Row():
            # Login button
            login_button = gr.Button("Login", variant="primary")
            # Register button
            register_button = gr.Button("Register", variant="secondary")
        # Login result output box (non-editable)
        login_output = gr.Textbox(label="Result", interactive=False)

    # Define chat page layout, initially hidden
    with gr.Column(visible=False) as chat_page:
        # Header layout with welcome text and buttons
        with gr.Row(elem_classes="header"):
            # Welcome text, initially empty
            welcome_text = gr.Markdown("### Welcome,")
            # Row layout for header buttons
            with gr.Row():
                # New conversation button
                new_conv_button = gr.Button("New Conversation", elem_classes="header-btn", variant="secondary")
                # History button
                history_button = gr.Button("History", elem_classes="header-btn", variant="secondary")
                # Logout button
                logout_button = gr.Button("Logout", elem_classes="header-btn", variant="secondary")

        # Chat area layout
        with gr.Column(elem_classes="chat-area"):
            # Display conversation title
            title_display = gr.Markdown("## Conversation Title", elem_id="title-display")
            # Chatbot dialogue box, height 450 px
            chatbot = gr.Chatbot(label="Chat", height=450)
            # Row layout for message input box and send button
            with gr.Row():
                # Message input box
                message = gr.Textbox(label="Message", placeholder="Type a message and press Enter", scale=8, container=False)
                # Send button
                send = gr.Button("Send", scale=2)

    # Define registration modal layout, initially hidden
    with gr.Column(visible=False, elem_classes="modal") as register_modal:
        # Registration username input box
        reg_username = gr.Textbox(label="Username", placeholder="Enter your username")
        # Registration password input box (hidden input)
        reg_password = gr.Textbox(label="Password", placeholder="Enter your password", type="password")
        # Row layout for submit and close buttons
        with gr.Row():
            # Submit registration button
            reg_button = gr.Button("Submit", variant="primary")
            # Close modal button
            close_button = gr.Button("Close", variant="secondary")
        # Registration result output box (non-editable)
        reg_output = gr.Textbox(label="Result", interactive=False)

    # Define history modal layout, initially hidden
    with gr.Column(visible=False, elem_classes="modal") as history_modal:
        # Display history title
        gr.Markdown("### Conversation History")
        # Dropdown for selecting a conversation, default option
        conv_dropdown = gr.Dropdown(label="Select Conversation", choices=["Please select a conversation"], value="Please select a conversation")
        # Load conversation button
        load_conv_button = gr.Button("Load Conversation", variant="primary")
        # Close history modal button
        close_history_button = gr.Button("Close", variant="secondary")

    # Define functions to show and hide registration modal
    def show_register_modal(): return gr.update(visible=True)
    def hide_register_modal(): return gr.update(visible=False)
    # Show history modal and update conversation list
    def show_history_modal(username): return gr.update(visible=True), gr.update(choices=get_conversation_list(username), value="Please select a conversation")
    # Hide history modal
    def hide_history_modal(): return gr.update(visible=False)
    # Logout function, reset all states
    def logout(): return False, None, None, gr.update(visible=True), gr.update(visible=False), "Logged out", [], None, [], "New Chat"
    # Update welcome text function
    def update_welcome_text(username): return gr.update(value=f"### Welcome, {username}")
    # Update title display function
    def update_title_display(title): return gr.update(value=f"## {title}")

    # Bind register button to show registration modal
    register_button.click(show_register_modal, None, register_modal)
    # Bind close button to hide registration modal
    close_button.click(hide_register_modal, None, register_modal)
    # Bind submit registration button to call registration function
    reg_button.click(register, [reg_username, reg_password], reg_output)

    # Bind login button to call login function
    login_button.click(
        login, [login_username, login_password], [logged_in, current_user, current_user_id, current_conversation, login_output]
    ).then(
        # Toggle page visibility based on login status
        lambda logged: (gr.update(visible=not logged), gr.update(visible=logged)), [logged_in], [login_page, chat_page]
    ).then(
        # Update welcome text
        update_welcome_text, [current_user], welcome_text
    ).then(
        # Load current conversation history
        lambda username, conv_id: users_db[username]["conversations"][conv_id]["history"] if username and conv_id else [],
        [current_user, current_conversation], chatbot_history
    ).then(
        # Update conversation title
        lambda username, conv_id: users_db[username]["conversations"][conv_id].get("title", "New Chat") if username and conv_id else "New Chat",
        [current_user, current_conversation], conversation_title
    ).then(
        # Update title display
        update_title_display, [conversation_title], title_display)

    # Bind logout button
    logout_button.click(
        logout, None, [logged_in, current_user, current_user_id, login_page, chat_page, login_output, chatbot, current_conversation, chatbot_history, conversation_title]
    )

    # Bind history button to show history modal
    history_button.click(show_history_modal, [current_user], [history_modal, conv_dropdown])
    # Bind close history button to hide history modal
    close_history_button.click(hide_history_modal, None, history_modal)

    # Bind new conversation button
    new_conv_button.click(new_conversation, [current_user], [login_output, current_conversation]
    ).then(
        # Clear chat dialog
        lambda: [], None, chatbot
    ).then(
        # Clear chat history state
        lambda: [], None, chatbot_history
    ).then(
        # Reset conversation title
        lambda: "New Chat", None, conversation_title
    ).then(
        # Update title display
        update_title_display, [conversation_title], title_display)

    # Bind load conversation button to load selected conversation history
    load_conv_button.click(load_conversation, [current_user, conv_dropdown], chatbot
    ).then(
        # Update current conversation ID
        lambda user, conv: extract_conversation_id(conv, user), [current_user, conv_dropdown], current_conversation
    ).then(
        # Update conversation title
        lambda username, conv: users_db[username]["conversations"][extract_conversation_id(conv, username)].get("title", "New Chat") if username and conv else "New Chat",
        [current_user, conv_dropdown], conversation_title
    ).then(
        # Update title display
        update_title_display, [conversation_title], title_display
    ).then(
        # Hide history modal
        hide_history_modal, None, history_modal)

    # Define function to update chat history
    def update_history(chatbot_output, history, user, conv_id):
        # If user and conversation ID exist, update chat history in the database
        if user and conv_id: users_db[user]["conversations"][conv_id]["history"] = chatbot_output
        return chatbot_output

    # Bind send button to send message and update interface
    send.click(
        send_message, [message, chatbot_history, current_user_id, current_conversation, current_user], [chatbot, chatbot_history, conversation_title]
    ).then(
        # Update chat history
        update_history, [chatbot, chatbot_history, current_user, current_conversation], chatbot_history
    ).then(
        # Update conversation title
        lambda username, conv_id: users_db[username]["conversations"][conv_id].get("title", "New Chat") if username and conv_id else "New Chat",
        [current_user, current_conversation], conversation_title
    ).then(
        # Update title display
        update_title_display, [conversation_title], title_display
    ).then(
        # Clear message input box
        lambda: "", None, message)

    # Bind message input submit (Enter key) to send message and update interface
    message.submit(
        send_message, [message, chatbot_history, current_user_id, current_conversation, current_user], [chatbot, chatbot_history, conversation_title]
    ).then(
        # Update chat history
        update_history, [chatbot, chatbot_history, current_user, current_conversation], chatbot_history
    ).then(
        # Update conversation title
        lambda username, conv_id: users_db[username]["conversations"][conv_id].get("title", "New Chat") if username and conv_id else "New Chat",
        [current_user, current_conversation], conversation_title
    ).then(
        # Update title display
        update_title_display, [conversation_title], title_display
    ).then(
        # Clear message input box
        lambda: "", None, message)

# If the script is run as main, launch the Gradio app
if __name__ == "__main__":
    # Launch Gradio app, listen on local port 7860
    demo.launch(server_name="127.0.0.1", server_port=7860)
