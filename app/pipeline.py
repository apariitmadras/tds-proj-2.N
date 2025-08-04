from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import asyncio, random
from google.genai.errors import ServerError

# ── 1. Gemini – planner ────────────────────────────────────────────────────
from google import genai                   # google-genai ≥ 1.x (v1 API)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
_GEMINI_CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ── 2. OpenAI – executor ───────────────────────────────────────────────────
from openai import AsyncOpenAI             # openai-python ≥ 1.x
OPENAI_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
_OPENAI_CLIENT = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Local helpers / schema ─────────────────────────────────────────────────
from .tools import scrape_website, get_relevant_data, answer_questions
from .models import AnswerPayload

# ───────────────────────────────────────────────────────────────────────────
# ❶  PLANNING  –  Gemini writes a step-by-step plan
# ───────────────────────────────────────────────────────────────────────────

async def make_plan_with_gemini(task: str) -> str:
    sys_prompt = ...
    for attempt in range(3):         # ≤3 tries: 0,1,2
        try:
            resp = _GEMINI_CLIENT.models.generate_content(
                model=GEMINI_MODEL,
                contents=[sys_prompt, task],
                generation_config=_GEN_CONFIG,
            )
            plan = resp.text.strip()
            Path("/tmp/breaked_task.txt").write_text(plan, encoding="utf-8")
            return plan
        except ServerError as e:
            if attempt == 2:          # last attempt – re-raise
                raise
            # jittered back-off: 1s, 2s
            await asyncio.sleep((2 ** attempt) + random.random())


# ───────────────────────────────────────────────────────────────────────────
# ❷  EXECUTION  –  GPT-4o-mini + function-calling tools
# ───────────────────────────────────────────────────────────────────────────
async def run_plan_with_gpt_tools(task: str) -> AnswerPayload:
    tools_schema: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "scrape_website",
                "description": "Download raw HTML and store it in a temp file.",
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

    # helper to execute a single tool call
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

    # initial request to GPT-4o-mini
    resp = await _OPENAI_CLIENT.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        tools=tools_schema,
        tool_choice="auto",
    )

    # ── main tool-execution loop ───────────────────────────────────────────
    while resp.choices[0].finish_reason == "tool_calls":
        assistant_msg = resp.choices[0].message
        messages.append(assistant_msg)  # keep assistant function-call message

        # handle each tool call and send matching tool reply
        for tc in assistant_msg.tool_calls:
            result = await _dispatch(tc)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,   # 🔑 required in openai-python ≥ 1.x
                    "content": result,
                }
            )

        # ask the model what’s next (or to finish)
        resp = await _OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
        )

    # ── final answer ──────────────────────────────────────────────────────
    final_json = json.loads(resp.choices[0].message.content)
    return AnswerPayload(answers=final_json)
