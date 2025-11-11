from typing import List, Optional
import os

from openai import AsyncOpenAI

from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric

DEFAULT_JUDGE_PROMPT = """Given a ground truth answer \
and a response, determine if the response is correct.

Question:
```
{question}
```

Ground truth answer:
```
{answer}
```

Response:
```
{response}
```

Respond either "yes" or "no" only."""


class JudgeRubric(Rubric):
    def __init__(
        self,
        parser: Parser = Parser(),
        parallelize_scoring: bool = False,
        judge_client: Optional[AsyncOpenAI] = None,
        judge_model: str = "gpt-4.1-nano",
        judge_sampling_args: dict = {},
        judge_prompt: str = DEFAULT_JUDGE_PROMPT,
        **kwargs,
    ):
        super().__init__(
            parser=parser, parallelize_scoring=parallelize_scoring, **kwargs
        )
        self.parser = parser
        # Allow routing judge calls to a vLLM/OpenAI-compatible endpoint via env vars.
        # Uses OPENAI_BASE_URL/OPENAI_API_KEY if provided (standard for OpenAI SDK),
        # falling back to JUDGE_BASE_URL/JUDGE_API_KEY. vLLM typically accepts any API key.
        if judge_client is not None:
            self.judge_client = judge_client
        else:
            base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("JUDGE_BASE_URL")
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("JUDGE_API_KEY") or "EMPTY"
            self.judge_client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=5)
        # Allow overriding judge model via env var for vLLM served model names
        self.judge_model = os.getenv("JUDGE_MODEL", judge_model)
        self.judge_prompt = judge_prompt
        self.judge_sampling_args = judge_sampling_args

    async def judge(self, prompt, completion, answer, state, **kwargs) -> str:
        if "judge_response" in state:
            return state["judge_response"]
        if isinstance(prompt, list):
            question = prompt[-1]["content"]
        else:
            question = prompt
        response = self.parser.parse_answer(completion)
        judge_prompt = self.judge_prompt.format(
            question=question, answer=answer, response=response
        )
        judge_response = await self.judge_client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            **self.judge_sampling_args,
        )
        judge_response = str(judge_response.choices[0].message.content)
        state["judge_response"] = judge_response
        return judge_response
