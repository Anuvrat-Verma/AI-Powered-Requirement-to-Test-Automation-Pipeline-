import ollama
import time
import os

MODEL = "llama3"
OUTPUT_DIR = "amazon_automation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

AGENT1_PROMPT = '''Role
automation test engineer

Goal
Create a user story for Amazon test automation in Gherkin format with detailed acceptance criteria and additional information.

Backstory
Test automation is critical for ensuring the reliability and scalability of Amazon's systems. User stories provide a clear understanding of the testing requirements and expected outcomes, enabling the development of robust automated test scripts. Using Gherkin format ensures clarity and consistency across teams.

Description
As an Automation Test Engineer, your task is to create a detailed user story for Amazon test automation in Gherkin format. Follow these instructions:
1. Identify the feature or functionality to be tested.
2. Define the user or stakeholder.
3. Specify the expected benefit.
4. Write in 'As a... I want... So that...' format.
5. Create a descriptive title.
6. Develop acceptance criteria in Gherkin.
7. Include priority, coverage, dependencies, tools.

OUTPUT FORMAT:
Title: [Descriptive title]
User Story:
As a [user/stakeholder]
I want [action/functionality]
So that [benefit/outcome]

Acceptance Criteria:
Scenario: [description]
Given [context]
When [action]
Then [outcome]

Additional Details:
- Priority: [High/Medium/Low]
- Test Coverage: [areas]
- Dependencies: [list]
- Tools/Frameworks: [Selenium, pytest etc.]
- Notes: [extra info]

Now create a fresh, high-quality user story for the following Amazon feature: '''

AGENT2_PROMPT = '''Role
automation test engineer

Goal
Convert the received Gherkin User Story into comprehensive, well-structured test cases.

Backstory
Test cases derived from Gherkin stories ensure complete coverage.

Description
Break down the User Story into detailed test cases (positive, negative, edge, boundary).

INSTRUCTIONS:
1. Cover every scenario in Acceptance Criteria.
2. Add extra negative/edge cases.
3. Use clear steps.

OUTPUT FORMAT (use exactly):

Test Cases for: [Title]

TC-001
Title: [short title]
Type: [Positive/Negative/Edge/Boundary]
Priority: [High/Medium/Low]
Preconditions:
- ...
Test Steps:
1. ...
2. ...
Expected Result:
- ...

(continue for all TC-XXX)

Additional Notes:
- Total test cases: X
- Coverage: ...

Now generate test cases for the following User Story:

'''

AGENT3_PROMPT = '''Role
automation test engineer

Goal
Convert the received test cases into clean, production-ready pytest + Selenium scripts using Page Object Model.

IMPORTANT: Generate FULL COMPLETE CODE. Never use placeholders like "..." or "# ... full test steps here". Every method must have real working Selenium code.

OUTPUT FORMAT (start directly with the code):

```python
# test_amazon_[feature].py
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# ====================== PAGE OBJECTS ======================
class HomePage:
    def __init__(self, driver):
        self.driver = driver
        self.search_bar = (By.ID, "twotabsearchtextbox")
        self.search_button = (By.ID, "nav-search-submit-button")

    def open(self):
        self.driver.get("https://www.amazon.in")

    def search(self, query):
        WebDriverWait(self.driver, 10).until(EC.visibility_of_element_located(self.search_bar)).send_keys(query)
        WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable(self.search_button)).click()

# ====================== TEST CLASS ======================
class TestAmazonFeature:
    @pytest.fixture(scope="class")
    def setup(self):
        options = webdriver.ChromeOptions()
        # options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        driver.maximize_window()
        yield driver
        driver.quit()

    # All test methods below are FULLY implemented with real code
    def test_successful_scenario(self, setup):
        driver = setup
        # real steps here - no placeholders

Generate the COMPLETE ready-to-run pytest + Selenium code for the following test cases:

'''

def run_agent(prompt, model):
    print(f"🤖 Running agent... (30-90 seconds)")
    start = time.time()
    response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
    print(f"Done in {time.time()-start:.1f}s")
    return response['message']['content'].strip()

def main():
    print("\n" + "="*70)
    print("AMAZON 3-AGENT PIPELINE - FULLY FIXED")
    print("="*70)

    model = input(f"Enter model (default {MODEL}): ").strip() or MODEL
    feature = input("\nAmazon feature (e.g. Add to Cart, Checkout, Login): ").strip() or "Search"

    print(f"\nFeature: {feature}\n")

    print("1️⃣ Agent 1 → User Story")
    user_story = run_agent(AGENT1_PROMPT + feature, model)
    with open(f"{OUTPUT_DIR}/1_user_story.txt", "w", encoding="utf-8") as f:
        f.write(user_story)
    print("💾 Saved 1_user_story.txt\n")

    print("2️⃣ Agent 2 → Test Cases")
    test_cases = run_agent(AGENT2_PROMPT + user_story, model)
    with open(f"{OUTPUT_DIR}/2_test_cases.txt", "w", encoding="utf-8") as f:
        f.write(test_cases)
    print("💾 Saved 2_test_cases.txt\n")

    print("3️⃣ Agent 3 → Pytest + Selenium Script")
    script = run_agent(AGENT3_PROMPT + test_cases, model)

    # SUPER SAFE CLEANING - no quote problems at all
    script = script.strip()
    script = script.strip()
    script = script.replace("```python", "").replace("```", "").strip()

    filename = f"test_amazon_{feature.lower().replace(' ', '_')}.py"
    with open(f"{OUTPUT_DIR}/{filename}", "w", encoding="utf-8") as f:
        f.write(script)

    print(f"💾 Final script → {OUTPUT_DIR}/{filename}")

if __name__ == "__main__":
    main()