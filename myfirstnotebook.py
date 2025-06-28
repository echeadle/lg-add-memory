import io
from typing import Annotated
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from PIL import Image as PILImage

from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file
load_dotenv(find_dotenv())

# Initialize the language model
llm = init_chat_model("openai:gpt-4o-mini")

def draw_graph(graph):
    """
    Generates and displays a diagram of the graph.
    """
    try:
        # Get the graph as a PNG image
        img_data = graph.get_graph().draw_mermaid_png()
        # Open the image and display it
        image = PILImage.open(io.BytesIO(img_data))
        image.show()
    except Exception as e:
        # This requires some extra dependencies and is optional.
        # Added error printing for better feedback.
        print(f"Error drawing graph: {e}")


class State(TypedDict):
    """
    Represents the state of our graph.
    """
    messages: Annotated[list, add_messages]


# Initialize the search tool and bind it to the LLM
tool = TavilySearch(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    """
    A node that invokes the chatbot to respond to the user's message.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Set up the memory saver
# This will persist the state of the graph, allowing for conversational memory
memory = MemorySaver()

# Define the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# Define the edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")

# Compile the graph with the memory saver
graph = graph_builder.compile(checkpointer=memory)


def stream_graph_updates(user_input: str, thread_id: str):
    """
    Streams the graph's response to the user's input for a specific conversation thread.
    """
    # The `configurable` dictionary is key to enabling memory.
    # The `thread_id` is used to track the conversation history.
    config = {"configurable": {"thread_id": thread_id}}
    
    # Stream the events from the graph
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config):
        for value in event.values():
            # Print the content of the last message, if it exists
            if value['messages'][-1].content:
                print("Assistant:", value['messages'][-1].content)


def main():
    """
    Main loop to run the chatbot from the command line.
    """
    print("Chatbot is ready. Type 'quit', 'exit', or 'q' to end.")
    print("Type 'draw' or 'graph' to see a diagram of the graph.")
    
    # A unique ID for the conversation thread
    thread_id = "my-first-conversation"
    print(f"Using conversation thread ID: {thread_id}")

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            elif user_input.lower() in ["draw", "graph", "show"]:
                print("Attempting to draw the graph...")
                draw_graph(graph)
                continue

            # Stream the graph updates for the user's input
            stream_graph_updates(user_input, thread_id)
        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C or Ctrl+D
            print("\nGoodbye!")
            break
        except Exception as e:
            # Fallback for other errors
            print(f"\nAn unexpected error occurred: {e}")
            break


if __name__ == "__main__":
    main()
