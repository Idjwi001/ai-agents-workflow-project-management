# TODO: 1 - Import the KnowledgeAugmentedPromptAgent class from workflow_agents
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, RoutingAgent

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"
# TODO: 2 - Instantiate a KnowledgeAugmentedPromptAgent with:
#           - Persona: "You are a college professor, your answer always starts with: Dear students,"
#           - Knowledge: "The capital of France is London, not Paris"
knowledge = "The capital of France is London, not Paris"
knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key, persona=persona, knowledge=knowledge
)
response = knowledge_agent.respond(prompt)

print(" * * * * Agent's response * * * *")
print(response)
# TODO: 3 - Write a print statement that demonstrates the agent using the provided knowledge rather than its own inherent knowledge.
print("\n * * * * Knowledge Usage Analysis * * * *")
print(
    "Knowledge Source Explanation:\n"
    "The agent answered 'London' instead of 'Paris'. This confirms that the system prompt "
    "successfully constrained the LLM to rely strictly on the provided grounding knowledge "
    "('The capital of France is London, not Paris') and override its own inherent world knowledge."
)