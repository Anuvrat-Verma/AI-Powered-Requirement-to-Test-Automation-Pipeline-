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


# Backend/agents.py

def get_targeted_context(agent_name: str, input_data: str) -> str:
    """Uses the LLM to generate a specific search query, then fetches from ChromaDB."""
    agent_role = config[agent_name]['role']

    # 1. Add steering instructions based on the agent's identity
    special_instruction = ""
    if agent_name == "automation_engineer":
        special_instruction = (
            "IMPORTANT: Do NOT query for business details. Focus exclusively on "
            "Browser setup, framework, architecture, assertions, Selenium POM standards, Python locator strategies (By.ID/By.CSS_SELECTOR), "
            "and WebDriverWait implementation rules."
        )
    elif agent_name == "qa_engineer":
        special_instruction = "Focus on edge cases, negative testing, and Gherkin-to-Test-Case mapping."

    # 2. Update the prompt to include the special instruction
    query_prompt = f"""You are a technical search assistant for a {agent_role}. 
{special_instruction}

Based on the input data, write a single, professional search query to retrieve coding standards, 
architectural templates, or testing rules from our knowledge base.

Input Data: 
{input_data}

Return ONLY the search string."""

    # 3. Generate the query
    search_query = llm.invoke(query_prompt).strip()

    # This will now print something like:
    # "Selenium POM locator standards CSS_SELECTOR WebDriverWait template"
    print(f"🔍 [{agent_role}] Optimized Query: {search_query}")

    # 4. Retrieve the specific documents
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


# Backend/agents.py

def agent3_generate_code(test_cases: str, use_rag: bool = False, feedback: str = "") -> str:
    """Now accepts an optional feedback string for self-correction."""
    context = get_targeted_context("automation_engineer", test_cases) if use_rag else ""

    # If there is feedback, we append it to the input so the LLM sees its mistakes
    input_data = test_cases
    if feedback:
        input_data = f"{test_cases}\n\n⚠️ PREVIOUS ATTEMPT FAILED COMPLIANCE:\n{feedback}\nPlease refactor the code to fix these issues."

    prompt = build_prompt("automation_engineer", input_data, context)
    return llm.invoke(prompt).strip()