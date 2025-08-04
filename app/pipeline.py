from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

# ─── Google Gemini (planner) ────────────────────────────────────────────────
from google import genai                                           # google-genai v1

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
_GEMINI_CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ─── OpenAI (executor) ──────────────────────────────────────────────────────
from openai import AsyncOpenAI                                      # openai-python ≥1
OPENAI_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
_OPENAI_CLIENT = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ─── Local tools & response schema ─────────────────────────────────────────
from .tools import scrape_website, get_relevant_data, answer_questions
from .models import AnswerPayload


# ────────────────────────────────────────────────────────────────────────────
# ❶  PLANNING STAGE  –  Gemini writes a step-by-step plan
# ────────────────────────────────────────────────────────────────────────────
async def make_plan_with_gemini(task: str) -> str:
    """Return a four-stage plan as plain text."""
    sys_prompt = (
        Path(__file__).with_name("prompts").joinpath("breakdown.txt").read_text()
    )

    resp = _GEMINI_CLIENT.models.generate_content(
        model=GEMINI_MODEL,
        contents=[sys_prompt, task])
    plan = resp.text.strip()
    # (optional) persist like the original repo
    Path("/tmp/breaked_task.txt").write_text(plan, encoding="utf-8")
    return plan


# ────────────────────────────────────────────────────────────────────────────
# ❷  EXECUTION STAGE  –  GPT-4o-mini + tool calls
# ────────────────────────────────────────────────────────────────────────────
async def run_plan_with_gpt_tools(task: str) -> AnswerPayload:
    """Run the analysis task end-to-end and return the final JSON answer."""
    tools_schema: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "scrape_website",
                "description": "Download raw HTML of a web page and save to a temp file.",
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
                "description": "Execute arbitrary Python code and return its stdout.",
                "parameters": {
                    "type": "object",
                    "properties": {"python_code": {"type": "string"}},
                    "required": ["python_code"],
                },
            },
        },
    ]

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are a smart data-analysis agent."},
        {"role": "user", "content": task},
    ]

    # helper to execute a single tool call ----------------------------------
    async def _dispatch(tool_call) -> str:
        fn_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        if fn_name == "scrape_website":
            return await scrape_website(**args)
        if fn_name == "get_relevant_data":
            return await get_relevant_data(**args)
        if fn_name == "answer_questions":
            return await answer_questions(**args)
        raise ValueError(f"Unknown tool {fn_name}")

    # first LLM call --------------------------------------------------------
    resp = await _OPENAI_CLIENT.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        tools=tools_schema,
        tool_choice="auto",
    )

    # tool-loop (linear, no recursion) --------------------------------------
    while resp.choices[0].finish_reason == "tool_calls":
        results = []
        for tc in resp.choices[0].message.tool_calls:
            results.append(await _dispatch(tc))
        messages.append(resp.choices[0].message)  # function call message
        messages.append(                         # tool response message
            {"role": "tool", "content": json.dumps(results)}
        )
        resp = await _OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
        )

    # final answer ----------------------------------------------------------
    final_json = json.loads(resp.choices[0].message.content)
    return AnswerPayload(answers=final_json)
