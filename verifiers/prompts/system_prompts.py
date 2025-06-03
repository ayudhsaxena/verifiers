SIMPLE_PROMPT = """
Respond in the following format, using careful step-by-step reasoning.

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

CODE_PROMPT = """\
Given a math problem, use step-by-step reasoning and code execution to solve the problem. 

For each step:
1. Think through your reasoning inside <reasoning> tags
2. Write Python scripts inside <code> tags to work out calculations
   - Functions and variables do not persist across <code> calls and should be redefined each time
   - Scripts should be written in Python 3.10+ syntax, and should run in under 10 seconds
   - Any desired outputs should be printed using print() statements
   - You may import numpy, scipy, and sympy libraries for your calculations
3. You will see the output from print() statements in your code in <output> tags
4. Continue until you can give the final answer inside <answer> tags
"""

DEFAULT_TOOL_PROMPT_TEMPLATE = """\
You have access to the following tools to help solve problems:

{tool_descriptions}

For each step:
1. Think through your reasoning inside <reasoning> tags
2. If needed, use a tool by writing a JSON command inside <tool> tags with:
   - "name": the tool to use
   - "args": the arguments for the tool
3. You will see the tool's output inside <result> tags
4. Continue until you can give the final answer inside <answer> tags

Tools expect specific JSON input formats. Follow the examples carefully.
Do not make up tools or arguments that aren't listed.
"""

TEXTARENA_PROMPT = """
You are a player in a game and your aim is to win the game. At every turn of yours, reason through different strategies inside the <reasoning> tags and choose the most optimal one. 
After reasoning, then output your response for the opponent within the <response> tags. Respond in the following format:
<reasoning>...</reasoning>
<response>...</response>
"""

TEXTARENA_PROMPT_V2 = """
You are a player in a game and your aim is to win the game. At every turn of yours, think through different strategies inside the <think> tags and choose the most optimal one. 
After thinking, then output your response for the opponent within the <answer> tags. Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""