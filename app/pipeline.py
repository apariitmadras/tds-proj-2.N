from __future__ import annotations

import asyncio, json, os, random
from pathlib import Path
from typing import Any, Dict, List

# ── 1. Gemini (planner) ────────────────────────────────────────────────────
from google import genai                          # google-genai ≥1.0
from google.genai.errors import ServerError

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
_GEMINI_CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ── 2. OpenAI (executor) ───────────────────────────────────────────────────
from openai import AsyncOpenAI                    # openai-python ≥1.0
OPENAI_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
_OPENAI_CLIENT = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Local helpers / schema ─────────────────────────────────────────────────
from .tools import scrape_website, get_relevant_data, answer_questions
from .models import AnswerPayload

# ───────────────────────────────────────────────────────────────────────────
# ❶  PLANNING  – Gemini writes a step-by-step plan
# ───────────────────────────────────────────────────────────────────────────
async def make_plan_with_gemini(task: str) -> str:
    sys_prompt = (
        Path(__file__).with_name("prompts").joinpath("breakdown.txt").read_text()
    )

    for attempt in range(3):                                   # retry ≤3×
        try:
            resp = _GEMINI_CLIENT.models.generate_content(
                model=GEMINI_MODEL,
                contents=[sys_prompt, task])
            plan = resp.text.strip()
            Path("/tmp/breaked_task.txt").write_text(plan, encoding="utf-8")
            return plan
        except ServerError:                                     # 503 etc.
            if attempt == 2:
                raise
            await asyncio.sleep((2**attempt) + random.random())  # 1 s, 2 s

# ───────────────────────────────────────────────────────────────────────────
# ❷  EXECUTION  – GPT-4o-mini + function-calling tools
# ───────────────────────────────────────────────────────────────────────────
async def run_plan_with_gpt_tools(task: str) -> AnswerPayload:
    tools_schema: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "scrape_website",
                "description": "Download raw HTML and save to a temp file.",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_relevant_data",
                "description": "Extract elements from an HTML file via CSS selector.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "html_file": {"type": "string"},
                        "selector": {"type": "string"},
                    },
                    "required": ["html_file", "selector"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "answer_questions",
                "description": "Run Python code and return its stdout.",
                "parameters": {
                    "type": "object",
                    "properties": {"python_code": {"type": "string"}},
                    "required": ["python_code"],
                },
            },
        },
    ]

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are an expert data-analysis agent."},
        {"role": "user", "content": task},
    ]

    async def _dispatch(call) -> str:
        fn = call.function.name
        args = json.loads(call.function.arguments)
        if fn == "scrape_website":
            return await scrape_website(**args)
        if fn == "get_relevant_data":
            return await get_relevant_data(**args)
        if fn == "answer_questions":
            return await answer_questions(**args)
        raise ValueError(f"unknown tool {fn}")

    resp = await _OPENAI_CLIENT.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        tools=tools_schema,
        tool_choice="auto",
    )

    # ── tool-execution loop ────────────────────────────────────────────────
    while resp.choices[0].finish_reason == "tool_calls":
        assistant_msg = resp.choices[0].message
        messages.append(assistant_msg)

        for tc in assistant_msg.tool_calls:
            result = await _dispatch(tc)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,         # required ≥1.0
                    "content": result,
                }
            )

        resp = await _OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
        )

    # ── final answer — retry once if not JSON ─────────────────────────────
    def _parse(s: str):
        try:  return json.loads(s)
        except json.JSONDecodeError:  return None

    answer = _parse(resp.choices[0].message.content)
    if answer is None:
        messages.append(resp.choices[0].message)
        messages.append(
            {
                "role": "system",
                "content": "⚠️ Previous reply was not valid JSON. "
                           "Respond again with ONLY the JSON array.",
            }
        )
        resp = await _OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
        )
        answer = _parse(resp.choices[0].message.content)
        if answer is None:
            raise ValueError("GPT final reply still not JSON.")

    return AnswerPayload(answers=answer)
