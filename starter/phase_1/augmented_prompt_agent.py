# TODO: 1 - Import the AugmentedPromptAgent class
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import AugmentedPromptAgent

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

# TODO: 2 - Instantiate an object of AugmentedPromptAgent with the required parameters
augmented_agent = AugmentedPromptAgent(
    openai_api_key=openai_api_key, persona=persona
)

# TODO: 3 - Send the 'prompt' to the agent and store the response in a variable named 'augmented_agent_response'
augmented_agent_response = augmented_agent.respond(prompt)

# Print the agent's response
print(augmented_agent_response)

# TODO: 4 - Add a comment explaining:
# - What knowledge the agent likely used to answer the prompt.
# - How the system prompt specifying the persona affected the agent's response.
print(
    "1. Knowledge Source:\n"
    "   The agent relied on general world knowledge embedded in the pre-trained weights "
    "of the gpt-3.5-turbo model, as no domain-specific ground truth context was provided.\n"
)

print(
    "2. Impact of Persona:\n"
    "   The system prompt forced the LLM to adopt a formal academic tone and adhere to "
    "the explicit structural constraint: starting the output with 'Dear students,'."
)