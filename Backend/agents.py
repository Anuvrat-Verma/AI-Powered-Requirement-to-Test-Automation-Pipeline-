import os
import yaml
from langchain_ollama import OllamaLLM

# Import the RAG retrieval function directly into the agents
from rag import retrieve_context

# 1. Safely locate and load the YAML file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "agents.yaml")

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)
ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# 2. Initialize Ollama
llm = OllamaLLM(model="llama3", temperature=0.3)


# 3. Dynamic Prompt Builder
def build_prompt(agent_name: str, input_data: str, context: str = "") -> str:
    """Dynamically builds the prompt from the YAML config."""
    agent = config[agent_name]
    context_block = f"\n--- COMPANY CONTEXT/GUIDELINES ---\n{context}\n----------------------------------\n" if context else ""

    return f"""You are a {agent['role']}.
{agent['goal']}
{context_block}
Input Data:
{input_data}

{agent['expected_output']}"""


# 4. Agentic RAG: The Query Generator
def get_targeted_context(agent_name: str, input_data: str) -> str:
    """Uses the LLM to generate a specific search query, then fetches from ChromaDB."""
    agent_role = config[agent_name]['role']

    # Ask Llama to write a search query based on what it's about to do
    query_prompt = f"""You are a database search assistant for a {agent_role}. 
Based on the following input, write a single, highly specific search query to find relevant rules, templates, or coding standards in our knowledge base.
Return ONLY the search string. Do not include conversational text or quotes.

Input Data: 
{input_data}"""

    # 1. Generate the query
    search_query = llm.invoke(query_prompt).strip()
    print(f"🔍 [{agent_role}] Querying database for: {search_query}")

    # 2. Retrieve the specific documents
    return retrieve_context(search_query)


# 5. Agent Execution Functions (Now with autonomous RAG)
def agent1_generate_stories(requirements: str, use_rag: bool = False) -> str:
    context = get_targeted_context("business_analyst", requirements) if use_rag else ""
    prompt = build_prompt("business_analyst", requirements, context)
    return llm.invoke(prompt).strip()


def agent2_generate_test_cases(stories: str, use_rag: bool = False) -> str:
    context = get_targeted_context("qa_engineer", stories) if use_rag else ""
    prompt = build_prompt("qa_engineer", stories, context)
    return llm.invoke(prompt).strip()


def agent3_generate_code(test_cases: str, use_rag: bool = False) -> str:
    context = get_targeted_context("automation_engineer", test_cases) if use_rag else ""
    prompt = build_prompt("automation_engineer", test_cases, context)
    return llm.invoke(prompt).strip()