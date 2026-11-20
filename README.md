# AI-Powered Agentic Workflow for Project Management

A small library of AI agents, and a workflow built from them, that turns a product
spec into a project plan: **user stories → product features → engineering tasks**.
The pilot input is the Email Router spec (`phase_2/Product-Spec-Email-Router.txt`).

## Project structure

```
phase_1/                              Agent library + one test script per agent
├── workflow_agents/base_agents.py    The 7 agent classes
├── direct_prompt_agent.py            Test: DirectPromptAgent
├── augmented_prompt_agent.py         Test: AugmentedPromptAgent
├── knowledge_augmented_prompt_agent.py  Test: KnowledgeAugmentedPromptAgent
├── rag_knowledge_prompt_agent.py     Test: RAGKnowledgePromptAgent (provided)
├── evaluation_agent.py               Test: EvaluationAgent
├── routing_agent.py                  Test: RoutingAgent
├── action_planning_agent.py          Test: ActionPlanningAgent
└── outputs/                          Terminal output of each test

phase_2/                              Project management workflow
├── workflow_agents/base_agents.py    Copy of the Phase 1 library
├── agentic_workflow.py               The workflow script
├── Product-Spec-Email-Router.txt     Input product spec
└── outputs/agentic_workflow_output.txt  Terminal output of the workflow
```

## The agents

| Agent | What it does |
|---|---|
| `DirectPromptAgent` | Sends the prompt straight to the model, no system prompt |
| `AugmentedPromptAgent` | Answers while playing a persona |
| `KnowledgeAugmentedPromptAgent` | Answers using only the knowledge it is given |
| `RAGKnowledgePromptAgent` | Finds the most relevant piece of a long text, then answers from it |
| `EvaluationAgent` | Checks another agent's answer against criteria and asks for fixes in a loop |
| `RoutingAgent` | Sends a prompt to the best-matching agent using embeddings + cosine similarity |
| `ActionPlanningAgent` | Breaks a request into a clean list of steps |

Chat agents use `gpt-4.1-mini`; embeddings use `text-embedding-3-large`.

## How the workflow works

1. The **Action Planning Agent** splits the request into steps.
2. The **Routing Agent** sends each step to the right team:
   Product Manager (user stories), Program Manager (features) or Development Engineer (tasks).
3. Each team is a **Knowledge Augmented Prompt Agent** (does the work) paired with an
   **Evaluation Agent** (checks the format, up to 10 rounds).
4. The script prints one combined project plan with the three sections.

## How to run

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` in `phase_1/` and `phase_2/` and add your key:

```
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://openai.vocareum.com/v1   # remove this line for the normal OpenAI API
```

Run the scripts from inside their folder:

```bash
cd phase_1 && python direct_prompt_agent.py
```

```bash
cd phase_2 && python agentic_workflow.py
```
