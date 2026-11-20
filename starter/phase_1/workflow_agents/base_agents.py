"""
Reusable agent library for agentic workflows.

Every agent receives the OpenAI API key when it is created (nothing is hardcoded).
The OpenAI client also reads the optional OPENAI_BASE_URL environment variable,
so the same code works with the normal OpenAI API or with a proxy such as Vocareum.
"""

# TODO: 1 - import the OpenAI class from the openai library
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime
from openai import OpenAI

# Model used by every chat agent in this library
CHAT_MODEL = "gpt-4.1-mini"
# Model used to turn text into embeddings (RoutingAgent and RAG agent)
EMBEDDING_MODEL = "text-embedding-3-large"


# DirectPromptAgent class definition
class DirectPromptAgent:
    """Sends the user's prompt straight to the LLM, with no system prompt or extra context."""

    def __init__(self, openai_api_key):
        # TODO: 2 - Define an attribute named openai_api_key to store the OpenAI API key provided to this class.
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        """Return the LLM's text answer to the prompt."""
        client = OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=CHAT_MODEL,  # TODO: 3 - Specify the model to use (gpt-4.1-mini)
            # TODO: 4 - Provide the user's prompt here. Do not add a system prompt.
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        # TODO: 5 - Return only the textual content of the response (not the full JSON response).
        return response.choices[0].message.content


# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    """Answers prompts while playing a given persona (set through the system prompt)."""

    def __init__(self, openai_api_key, persona):
        """Store the API key and the persona the agent should adopt."""
        # TODO: 1 - Create an attribute for the agent's persona
        self.persona = persona
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Return the LLM's text answer, written in the agent's persona."""
        client = OpenAI(api_key=self.openai_api_key)

        # TODO: 2 & 3 - System prompt with persona instruction + user message.
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"{self.persona}. Forget all previous context.",
                },
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )

        # TODO: 4 - Return only the textual content of the response, not the full JSON payload.
        return response.choices[0].message.content


# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    """Answers prompts in a persona, using ONLY the knowledge it was given (not the model's own)."""

    def __init__(self, openai_api_key, persona, knowledge):
        """Store the API key, the persona and the knowledge the agent must rely on."""
        self.persona = persona
        # TODO: 1 - Create an attribute to store the agent's knowledge.
        self.knowledge = knowledge
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Return the LLM's text answer, based only on self.knowledge."""
        client = OpenAI(api_key=self.openai_api_key)

        # TODO: 2 - System message with the persona, the knowledge and the final instruction.
        system_prompt = (
            f"You are {self.persona} knowledge-based assistant. Forget all previous context.\n"
            f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}\n"
            "Answer the prompt based on this knowledge, not your own."
        )

        # TODO: 3 - Add the user's input prompt here as a user message.
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# RAGKnowledgePromptAgent class definition (provided)
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(api_key=self.openai_api_key)
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            chunks = [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]
        else:
            chunks, start, chunk_id = [], 0, 0

            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                if separator in text[start:end]:
                    end = start + text[start:end].rindex(separator) + len(separator)

                chunks.append({
                    "chunk_id": chunk_id,
                    "text": text[start:end],
                    "chunk_size": end - start,
                    "start_char": start,
                    "end_char": end
                })

                # Stop once the end of the text is reached, otherwise the overlap
                # would make the loop repeat the last chunk forever.
                if end >= len(text):
                    break
                start = end - self.chunk_overlap
                chunk_id += 1

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        client = OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content


class EvaluationAgent:
    """
    Checks a worker agent's answer against criteria, in a loop.

    Each round: the worker answers -> the evaluator says Yes/No -> if No, the
    evaluator writes fix instructions and the worker tries again. The loop stops
    when the answer is accepted or after max_interactions rounds.
    """

    def __init__(
        self,
        openai_api_key: str,
        persona: str,
        evaluation_criteria: str,
        worker_agent,
        max_interactions: int = 3,
    ):
        # TODO: 1 - Declare class attributes here
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        """Run the evaluate-and-correct loop and return the final result as a dictionary."""
        client = OpenAI(api_key=self.openai_api_key)
        prompt_to_evaluate = initial_prompt
        response_from_worker = ""
        evaluation = ""
        iterations = 0

        # TODO: 2 - Set loop to iterate up to the maximum number of interactions
        for i in range(self.max_interactions):
            iterations = i + 1
            print(f"\n--- Interaction {iterations} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            # TODO: 3 - Obtain a response from the worker agent
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"  # TODO: 4 - Insert evaluation criteria here
                "Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            # TODO: 5 - Define the message structure sent to the LLM for evaluation (use temperature=0)
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": self.persona},
                    {"role": "user", "content": eval_prompt},
                ],
                temperature=0,
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                break

            print(" Step 4: Generate instructions to correct the response")
            instruction_prompt = (
                f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
            )
            # TODO: 6 - Message structure to generate correction instructions (use temperature=0)
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": self.persona},
                    {"role": "user", "content": instruction_prompt},
                ],
                temperature=0,
            )
            instructions = response.choices[0].message.content.strip()
            print(f"Instructions to fix:\n{instructions}")

            print(" Step 5: Send feedback to worker agent for refinement")
            prompt_to_evaluate = (
                f"The original prompt was: {initial_prompt}\n"
                f"The response to that prompt was: {response_from_worker}\n"
                "It has been evaluated as incorrect.\n"
                f"Make only these corrections, do not alter content validity: {instructions}"
            )

        # TODO: 7 - Return a dictionary containing the final response, evaluation, and number of iterations
        return {
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "iterations": iterations,
        }


class RoutingAgent:
    """
    Sends a prompt to the best-matching agent.

    It turns the prompt and each agent's description into embeddings (lists of
    numbers that capture meaning), compares them with cosine similarity, and
    calls the agent whose description is the closest match.
    """

    def __init__(self, openai_api_key: str, agents: list):
        self.openai_api_key = openai_api_key
        # TODO: 1 - Define an attribute to hold the agents, call it agents
        # Each agent is a dict: {"name": ..., "description": ..., "func": callable}
        self.agents = agents

    def get_embedding(self, text):
        """Return the embedding vector of the text (text-embedding-3-large)."""
        client = OpenAI(api_key=self.openai_api_key)
        # TODO: 2 - Calculate the embedding of the text using the text-embedding-3-large model
        response = client.embeddings.create(
            model=EMBEDDING_MODEL, input=text, encoding_format="float"
        )
        return response.data[0].embedding

    # TODO: 3 - Define a method to route user prompts to the appropriate agent
    def route(self, user_input: str):
        """Pick the agent whose description is most similar to user_input and return its response."""
        # TODO: 4 - Compute the embedding of the user input prompt
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            # TODO: 5 - Compute the embedding of the agent description
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue

            # Cosine similarity: closer to 1 means more similar meaning
            similarity = np.dot(input_emb, agent_emb) / (
                np.linalg.norm(input_emb) * np.linalg.norm(agent_emb)
            )
            print(f"[Router] {agent['name']}: similarity = {similarity:.3f}")

            # TODO: 6 - Keep the agent with the highest similarity score
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)


class ActionPlanningAgent:
    """Breaks a request into a clean list of steps, using only the steps found in its knowledge."""

    def __init__(self, openai_api_key: str, knowledge: str):
        # TODO: 1 - Initialize the agent attributes here
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt: str) -> list:
        """Return the steps needed to complete the prompt, as a list of strings."""
        # TODO: 2 - Instantiate the OpenAI client using the provided API key
        client = OpenAI(api_key=self.openai_api_key)

        # TODO: 3 - Call the OpenAI API (gpt-4.1-mini) with the system prompt and the user's prompt
        system_prompt = (
            "You are an action planning agent. Using your knowledge, you extract from the "
            "user prompt the steps requested to complete the action the user is asking for. "
            "You return the steps as a list. Only return the steps in your knowledge. "
            f"Forget any previous context. This is your knowledge: {self.knowledge}"
        )
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        # TODO: 4 - Extract the response text from the OpenAI API response
        response_text = response.choices[0].message.content.strip()

        # TODO: 5 - Clean and format the extracted steps by removing empty lines and unwanted text
        steps = []
        for line in response_text.split("\n"):
            line = line.strip()
            if not line:
                continue  # skip empty lines
            # Remove list markers such as "1.", "2)", "-", "*", "•" and "Step 1:"
            line = re.sub(r"^(step\s*\d+\s*[:.)-]\s*|\d+\s*[.)]\s*|[-*•]\s*)", "", line, flags=re.IGNORECASE)
            line = line.replace("**", "").strip()
            # Skip intro/heading lines like "Here are the steps:" that are not real steps
            if not line or line.endswith(":"):
                continue
            steps.append(line)

        return steps
