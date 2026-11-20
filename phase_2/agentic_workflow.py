# agentic_workflow.py
"""
Agentic workflow for technical project management (pilot: Email Router).

How it works, in plain words:
  1. The Action Planning Agent splits the TPM's request into steps.
  2. The Routing Agent sends each step to the right "team":
       - Product Manager      -> writes user stories
       - Program Manager      -> groups stories into product features
       - Development Engineer -> writes engineering tasks
  3. Each team = a Knowledge Augmented Prompt Agent (does the work) paired with an
     Evaluation Agent (checks the format and asks for fixes until it is right).
  4. At the end, all results are printed as one project plan.
"""

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------

# TODO: 1 - Import the following agents: ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent from the workflow_agents.base_agents module
import os
from dotenv import find_dotenv, load_dotenv

from workflow_agents.base_agents import (
    ActionPlanningAgent,
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
    RoutingAgent,
)

# TODO: 2 - Load the OpenAI key into a variable called openai_api_key
load_dotenv(find_dotenv(usecwd=True))
openai_api_key = os.getenv("OPENAI_API_KEY")

# TODO: 3 - Load the product spec document Product-Spec-Email-Router.txt into a variable called product_spec
product_spec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Product-Spec-Email-Router.txt")
with open(product_spec_path, "r", encoding="utf-8") as spec_file:
    product_spec = spec_file.read()

# Where each team's validated output is collected, so we can print one full plan at the end
project_plan = {
    "user_stories": [],
    "product_features": [],
    "engineering_tasks": [],
}

# ---------------------------------------------------------------------------
# 2. Instantiate all the agents
# ---------------------------------------------------------------------------

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
# TODO: 4 - Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(
    openai_api_key=openai_api_key,
    knowledge=knowledge_action_planning,
)

# --- Product Manager team ---------------------------------------------------
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    # TODO: 5 - Complete this knowledge string by appending the product_spec loaded in TODO 3
    + product_spec
)
# TODO: 6 - Instantiate a product_manager_knowledge_agent using 'persona_product_manager' and the completed 'knowledge_product_manager'
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager,
    knowledge=knowledge_product_manager,
)

# TODO: 7 - Define the persona and evaluation criteria for a Product Manager evaluation agent and instantiate it
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria_product_manager = (
    "The answer should be stories that follow the following structure: "
    "As a [type of user], I want [an action or feature] so that [benefit/value]."
)
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager_eval,
    evaluation_criteria=evaluation_criteria_product_manager,
    worker_agent=product_manager_knowledge_agent,  # the agent to evaluate
    max_interactions=10,
)

# --- Program Manager team ---------------------------------------------------
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
# Each step is routed on its own, so the Program Manager would not otherwise know
# which product it is planning. Adding the spec keeps its features about the Email Router.
knowledge_program_manager += "\nThis is the product spec the features are for:\n" + product_spec

# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager,
)

# TODO: 8 - Instantiate a program_manager_evaluation_agent using 'persona_program_manager_eval' and the evaluation criteria below.
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_program_manager = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=evaluation_criteria_program_manager,
    worker_agent=program_manager_knowledge_agent,  # the agent to evaluate
    max_interactions=10,
)

# --- Development Engineer team ----------------------------------------------
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
# Same reason as above: give the engineer the spec so the tasks are about the Email Router.
knowledge_dev_engineer += "\nThis is the product spec the tasks are for:\n" + product_spec

# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer,
)

# TODO: 9 - Instantiate a development_engineer_evaluation_agent using 'persona_dev_engineer_eval' and the evaluation criteria below.
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_dev_engineer = (
    "The answer should be tasks following this exact structure: "
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=evaluation_criteria_dev_engineer,
    worker_agent=development_engineer_knowledge_agent,  # the agent to evaluate
    max_interactions=10,
)

# ---------------------------------------------------------------------------
# 3. Support functions (one per team)
# ---------------------------------------------------------------------------
# TODO: 11 - Each support function takes a step from the action plan, passes the step
# itself to its Evaluation Agent (which calls its own worker agent internally),
# and returns the 'final_response' from the evaluation result.

def product_manager_support_function(query: str) -> str:
    """Product Manager team: write user stories for the step and return the validated result."""
    result = product_manager_evaluation_agent.evaluate(query)
    project_plan["user_stories"].append(result["final_response"])
    return result["final_response"]


def program_manager_support_function(query: str) -> str:
    """Program Manager team: define product features for the step and return the validated result."""
    result = program_manager_evaluation_agent.evaluate(query)
    project_plan["product_features"].append(result["final_response"])
    return result["final_response"]


def development_engineer_support_function(query: str) -> str:
    """Development Engineer team: define engineering tasks for the step and return the validated result."""
    result = development_engineer_evaluation_agent.evaluate(query)
    project_plan["engineering_tasks"].append(result["final_response"])
    return result["final_response"]


# ---------------------------------------------------------------------------
# 4. Routing Agent
# ---------------------------------------------------------------------------
# TODO: 10 - Instantiate a routing_agent and give it one route per team.
routing_agent = RoutingAgent(openai_api_key, [])

agents_routes = [
    {
        "name": "Product Manager",
        "description": (
            "Responsible for defining product personas and user stories only. "
            "Does not define features or tasks. Does not group stories."
        ),
        "func": lambda x: product_manager_support_function(x),
    },
    {
        "name": "Program Manager",
        "description": (
            "Responsible for defining product features by grouping related user stories. "
            "Does not write user stories or engineering tasks."
        ),
        "func": lambda x: program_manager_support_function(x),
    },
    {
        "name": "Development Engineer",
        "description": (
            "Responsible for defining the engineering development tasks needed to build "
            "each user story. Does not write user stories or define features."
        ),
        "func": lambda x: development_engineer_support_function(x),
    },
]
routing_agent.agents = agents_routes

# ---------------------------------------------------------------------------
# 5. Run the workflow
# ---------------------------------------------------------------------------
print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "What would the development tasks for this product be?"
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
# TODO: 12 - Implement the workflow.
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
print("Workflow steps:")
for number, step in enumerate(workflow_steps, start=1):
    print(f"  {number}. {step}")

completed_steps = []

for number, step in enumerate(workflow_steps, start=1):
    print(f"\n========== Executing Step {number}: {step} ==========")

    # The router picks the right team and returns that team's validated answer
    step_result = routing_agent.route(step)
    completed_steps.append(step_result)

    print(f"\n[Result of Step {number}]:\n{step_result}\n")

# ---------------------------------------------------------------------------
# 6. Print one consolidated project plan
# ---------------------------------------------------------------------------
def print_section(title: str, results: list) -> None:
    """Print one labelled section of the project plan."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    if results:
        print("\n\n".join(results))
    else:
        print("(No output was produced for this section.)")


print("\n\n" + "#" * 70)
print("FINAL PROJECT PLAN - EMAIL ROUTER")
print("#" * 70)
print(f"Workflow steps completed: {len(completed_steps)} of {len(workflow_steps)}")
print_section("SECTION 1: USER STORIES (Product Manager)", project_plan["user_stories"])
print_section("SECTION 2: PRODUCT FEATURES (Program Manager)", project_plan["product_features"])
print_section("SECTION 3: ENGINEERING TASKS (Development Engineer)", project_plan["engineering_tasks"])
print("\n*** Workflow execution finished ***")
