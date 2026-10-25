import re
import shlex
import subprocess
from pathlib import Path
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages 
from langgraph.graph import StateGraph , START , END 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call

from langchain_chroma import Chroma
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv 
from embeddings import LocalEmbeddings

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent


#Step 1 - Building the RAG retrievers 
embeddings = LocalEmbeddings()

DB_PATH = str(PROJECT_ROOT / "Chroma_DB")
vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings,
    collection_name="linux_documents",
)
doc_retriever = vectorstore.as_retriever(search_kwargs = {"k":4})

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)
llm2 = ChatMistralAI(
    model="ministral-3b-2512"
)

READ_ONLY_COMMANDS = {
    "cat", "df", "du", "find", "free", "grep", "head", "ls", "lsof",
    "mount", "ps", "pwd", "stat", "top", "uptime", "uname", "whoami",
}
BLOCKED_TOKENS = {";", "&&", "||", "|", ">", ">>", "<", "`", "$()"}
BLOCKED_ARGUMENTS = {"-exec", "-execdir", "-delete", "--exec", "--execdir"}


@tool
def terminal(command: str) -> str:
    """Run one approved read-only Linux command in the local terminal."""
    try:
        command_parts = shlex.split(command)
    except ValueError as error:
        return f"Invalid command syntax: {error}"

    if not command_parts:
        return "No command was provided."
    if any(token in command for token in BLOCKED_TOKENS):
        return "Rejected: shell operators, pipes, redirection, and substitutions are not allowed."

    executable = Path(command_parts[0]).name
    if executable not in READ_ONLY_COMMANDS:
        return f"Rejected: '{executable}' is not an allowed read-only command."
    if any(argument in BLOCKED_ARGUMENTS for argument in command_parts[1:]):
        return "Rejected: this command contains an operation that can modify the system."

    try:
        result = subprocess.run(
            command_parts,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"Command could not be completed: {error}"

    output = result.stdout.strip() or result.stderr.strip()
    if len(output) > 12000:
        output = output[:12000] + "\n[Output truncated]"
    return output or f"Command exited with code {result.returncode}."


@wrap_tool_call
def human_approval(request, handler):
    """Require explicit approval before every tool call."""
    tool_call = request.tool_call
    print(f"Agent requests tool '{tool_call['name']}' with {tool_call['args']}")
    if input("Approve this tool call? (yes/no): ").strip().lower() != "yes":
        return ToolMessage(
            content="The user rejected this tool call.",
            tool_call_id=tool_call["id"],
        )
    return handler(request)


pc_agent = create_agent(
    llm,
    tools=[terminal],
    system_prompt=(
        "You are a Linux PC diagnostic assistant running locally on the user's computer. "
        "You MUST call the terminal tool for questions about the current computer; "
        "never claim that you cannot access the filesystem. Use read-only commands such "
        "as pwd, ls, df -h, du, find, free -h, ps, and uname -a. For disk-space questions, "
        "inspect both filesystem usage and large files. Never use destructive commands, "
        "sudo, or commands that modify files. Never claim a command succeeded without its result."
    ),
    middleware=[human_approval],
)


def response_text(response) -> str:
    """Extract plain text from a LangChain response or structured content."""
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)

    return str(content)

#step2 - State 

class State(TypedDict):
    prompt : str
    messages : Annotated[list,add_messages]
    conversation_history: list[dict[str, str]]
    query_type : str
    needs_tools : bool
    retrieved_context : str
    tool_context : str
    final_op : str 
    review_feedback : str
    is_approved : bool
    attempt : int

#Step 3 - Nodes generation 
    
def classifier_node(state : State) -> dict:
    """Classify the user's prompt into the documentation or general path."""

    prompt = (
        "Classify the following query into exactly one category: "
        "either 'documentation' or 'no_documentation'.\n\n"
        "Use 'documentation' for questions that require information "
        "from the Linux documentation, system manuals, troubleshooting guides, "
        "configuration documentation, or stored runbooks.\n"
        "Use 'no_documentation' for questions that can be answered directly "
        "from the current system state, user-provided information, or by running "
        "available system commands/tools.\n\n"
        f"Query: {state['prompt']}\n\n"
        "Also return TOOLS=yes when current computer information or an editable "
        "operation is needed, otherwise TOOLS=no. Format: CATEGORY=<category> TOOLS=<yes/no>."
    )

    response = llm.invoke(prompt)
    classification = response_text(response).strip().lower()

    if "no_documentation" in classification:
        category = "no_documentation"
    else:
        category = "documentation"

    needs_tools = bool(re.search(r"tools\s*[=:]\s*yes", classification))
    if any(word in state["prompt"].lower() for word in
           ("current", "my computer", "disk", "memory", "ram", "directory", "file")):
        needs_tools = True
    
    return {"query_type" : category, "needs_tools": needs_tools}

def rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the linux documentation."""
    query = state["prompt"]
    docs = doc_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {
        "retrieved_context": context,
        "tool_context": run_pc_tools(state) if state["needs_tools"] else "",
    }


def general_node(state: State) -> dict:
    """Gathers live computer information when the question requires it."""
    return {
        "retrieved_context": "NO_RETRIEVAL_NEEDED",
        "tool_context": run_pc_tools(state) if state["needs_tools"] else "",
    }


def run_pc_tools(state: State) -> str:
    """Ask the tool-enabled agent for current computer information."""
    tool_prompt = (
        f"User request: {state['prompt']}\n\n"
        "You are operating on the user's local Linux computer. Use the terminal tool now when "
        "the request needs current system information. For disk-space questions, "
        "run commands such as 'df -h' and 'du -ah' or 'find' to inspect large files. "
        "Do not answer that you lack filesystem access."
    )
    result = pc_agent.invoke({"messages": [("human", tool_prompt)]})
    return response_text(result["messages"][-1])

def reviewer_node(state:State) -> dict:
    """Reviews the draft and decides: approve or reject with feedback."""
    draft = state['final_op']
    user_prompt = state['prompt']

    review_prompt = (
        f"Review this answer with respect to the user's prompt: {user_prompt}\n\n"
        f"{draft}\n"
        "Return APPROVED if it is correct and complete. Otherwise return "
        "REJECTED and explain what must be corrected using FEEDBACK:."
    )
    response = llm2.invoke(
        [("system", "Respond with APPROVED or REJECTED, with FEEDBACK when rejected."),
         ("human", review_prompt)]
    )
    review_text = response_text(response).strip()
    
    verdict_text = review_text.split("feedback", 1)[0].upper()
    is_approved = "APPROVED" in verdict_text and "REJECTED" not in verdict_text

    feedback_match = re.search(r"feedback\s*:\s*(.*)", review_text, re.IGNORECASE | re.DOTALL)
    if feedback_match:
        feedback = feedback_match.group(1).strip()
    else:
        feedback = review_text

    verdict = "APPROVED" if is_approved else "REJECTED"
    print(f"[Verdict: {verdict}]")
    print(f"[Feedback: {feedback}]")

    return {
        "review_feedback": feedback,
        "is_approved": is_approved,
    }

def draft_node(state : State) -> dict:
    """Generates the response to user query."""
    attempt = state.get("attempt",0) + 1 
    question = state["prompt"]
    previous_feedback = state.get('review_feedback', '')
    context = state.get("retrieved_context", "")
    tool_context = state.get("tool_context", "")
    history = state.get("conversation_history", [])
    history_text = "\n\n".join(
        f"User: {item['prompt']}\nAssistant: {item['response']}"
        for item in history[-10:]
    )
    if attempt == 1:
        user_message = (
            f"Previous conversation:\n{history_text or 'No previous conversation.'}\n\n"
            f"User question: {question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"Current computer information:\n{tool_context}\n\n"
            "Answer the question clearly. Use the retrieved context when it is available."
        )
    else:
        user_message = (
            f"The previous answer to '{question}' was rejected.\n\n"
            f"Reviewer's feedback:\n{previous_feedback}\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"Current computer information:\n{tool_context}\n\n"
            "Write a corrected answer using the context when available. Do not repeat the same mistake."
        )
    response = llm.invoke([("human", user_message)])
    answer = response_text(response).strip()

    return {
        "messages" : [("ai", answer)],
        "final_op" : answer,
        "attempt" : attempt
    }

#step 4 - router function 

def route_query(state:State):
    if state['query_type'] == 'no_documentation':
        return "general"
    else:
        return "rag"

def should_stop_looping(state:State):
    if state['is_approved']:
        print("ans has been approved \n")
        return END
    if state['attempt'] >= 1:
        print("reached max attempts")
        return END 
    return "draft"

#step 5 - Building the graph 

graph = StateGraph(State)

graph.add_node("classifier",classifier_node)
graph.add_node("rag",rag_node)
graph.add_node("general",general_node)
graph.add_node("review",reviewer_node)
graph.add_node("draft",draft_node)

#edges 

graph.add_edge(START,"classifier")

graph.add_conditional_edges(
    "classifier",route_query
)

graph.add_edge("rag","draft")
graph.add_edge("general","draft")

graph.add_edge("draft","review")

graph.add_conditional_edges(
    "review",should_stop_looping
)

app = graph.compile()
#step 6 - Run the code 

if __name__ == "__main__":
    print("Yours pc expert \n\n")

    conversation_history = []

    while True:
        user_query = input("You:  ")

        if user_query.lower() in ["exit", "quit"]:
            break

        result = app.invoke({
            "prompt": user_query,
            "messages": [("human", user_query)],
            "conversation_history": conversation_history,
            "retrieved_context": "",
            "final_op": "",
            "review_feedback": "",
            "is_approved": False,
            "attempt": 0,
            "query_type": "",
            "needs_tools": False,
            "tool_context": "",
        })

        final_answer = result["messages"][-1].content
        conversation_history.append({
            "prompt": user_query,
            "response": final_answer,
        })
        #conversation_history = conversation_history[-10:]

        print(f"Assistant: {final_answer}")