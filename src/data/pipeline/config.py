"""Configuration constants for LCoT2Tree Pipeline

Contains:
- SPLIT_WORDS: Markers for thought segmentation
- PROMPTS: Templates for each pipeline step
- CATEGORY_*: Relationship type definitions
"""

# ============================================================================
# THOUGHT SEGMENTATION
# ============================================================================

SPLIT_WORDS = [
    "Alternatively", "Wait, no", "Hmm", "But wait", "Let me verify",
    "let's verify", "Or wait", "To verify", "Wait", "Verify",
    "Let's confirm", "Let's check", "Another example", "But let's",
    "No:", "no:"
]


# ============================================================================
# RELATIONSHIP CATEGORIES
# ============================================================================

CATEGORY_MAP = {
    "Continuous Logic": 1,
    "Exploration": 2,
    "Backtracking": 3,
    "Validation": 4,
    "Default": 5,  # Structural fallback
}

CATEGORY_NAMES = {
    0: 'Root',
    1: 'Continuous Logic',
    2: 'Exploration',
    3: 'Backtracking',
    4: 'Validation',
    5: 'Default',
}


# ============================================================================
# STEP 2: REASONING SKETCH EXTRACTION
# ============================================================================

STEP2_SKETCH_PROMPT = """Analyze the following reasoning text and extract a strictly ordered, atomic sequence of key reasoning steps. Focus on extracting the validated, logically essential progression of thoughts while excluding backtracking, rechecks, or redundant details.

Reasoning text: 
<reasoning_text>
{text}
</reasoning_text>

Please read the entire text carefully and generate by following these rules:
1. Find the key steps and the logical flow of reasoning.
2. Each step must represent a single, indivisible logical action that directly advances the reasoning.
3. Determine the correct version of the step, ignoring redundant information. A correct step should be able to push the reasoning logic forward and have no errors in itself.
4. Do not skip steps. Do not merge steps. Use the original phrasing where possible.
5. Do not include verification steps unless it introduces new constraints.
6. Organize the steps into a coherent sequence of key reasoning steps and number it sequentially (1., 2., 3., ...).
7. Maintain strict output format.

Output format:
<reasoning_process>
Step 1. [concise statement]: [Details]
Step 2. [concise statement]: [Details]
Step 3. [concise statement]: [Details]
...
</reasoning_process>

Please list the key reasoning steps of the provided text.
"""


# ============================================================================
# STEP 3: THOUGHT-TO-STEP ASSIGNMENT
# ============================================================================

STEP3_SYSTEM_MESSAGE = """Your task is to assign specific thoughts to reasoning steps.

You will be given:
1. A numbered reasoning sketch (Step 1, Step 2, etc.)
2. A list of specific thoughts (Thought 0, Thought 1, etc.)

For each thought, determine which reasoning step(s) it belongs to based on:
- Semantic relevance
- Logical progression
- Content overlap

Guidelines:
- A thought can belong to multiple steps if it spans reasoning boundaries
- Consider the full context of both the thought and the step
- Be precise in your assignments

Output Format:
```json
{
  "Thought 0": [1, 2],
  "Thought 1": [2],
  "Thought 2": [3],
  ...
}
```"""


# ============================================================================
# STEP 4: PARENT RELATIONSHIP ASSIGNMENT
# ============================================================================

STEP4_SYSTEM_MESSAGE = """You are analyzing the thought dependencies in a chain of reasoning.

For the Current Thought from reasoning step N, you will be given a list of Candidate Parents from the previous logical step.
Your task: Identify which candidates are directly related to the current thought.

Relationship Categories:
1. Continuous Logic - Current thought directly continues or extends the parent's reasoning
2. Exploration - Current thought branches into alternative paths from the parent
3. Backtracking - Current thought revises or corrects the parent
4. Validation - Current thought validates or provides evidence for the parent

Guidelines:
- You MUST select at least ONE parent (the most relevant candidate)
- A thought can have MULTIPLE parents if there are multiple clear connections
- Select ALL parents with CLEAR, DIRECT connections

IMPORTANT: Always return at least one parent. Every thought builds on prior reasoning.

Output Format:
```json
{
  "parents": [
    {"id": <candidate_id>, "category": "<category_name>"},
    ...
  ]
}
```"""


# ============================================================================
# PIPELINE SETTINGS
# ============================================================================

# Maximum number of parallel workers for LLM calls
DEFAULT_MAX_WORKERS = 50

# Batch size for async processing
DEFAULT_BATCH_SIZE = 10

# Maximum text length for thought truncation (in characters)
MAX_THOUGHT_LENGTH = 500
MAX_CANDIDATE_LENGTH = 300
MAX_CURRENT_THOUGHT_DISPLAY = 200

# Temperature settings
DEFAULT_TEMPERATURE = 0.6
DEFAULT_MAX_TOKENS = 4096