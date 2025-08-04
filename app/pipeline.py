import os, asyncio, json
from typing import List
from pathlib import Path

from google import genai
import openai

from .tools import scrape_website, get_relevant_data, answer_questions
from .models import AnswerPayload

# ---------------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
async def make_plan_with_gemini(task: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    sys_prompt = Path(__file__).with_name("prompts").joinpath("breakdown.txt").read_text()

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[sys_prompt, task])
    plan = resp.text.strip()
    Path("/tmp/breaked_task.txt").write_text(plan, encoding="utf-8")
    return plan

# ---------------------------------------------------------------------------
async def run_plan_with_gpt_tools(task: str) -> AnswerPayload:
    openai.api_key = os.environ["OPENAI_API_KEY"]

    tools_schema = [
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
                "description": "Parse HTML file and extract elements matching a CSS selector.",
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
                "description": "Execute analysis Python and return its stdout.",
                "parameters": {
                    "type": "object",
                    "properties": {"python_code": {"type": "string"}},
                    "required": ["python_code"],
                },
            },
        },
    ]

    # ---- initial user message ----
    messages = [
        {"role": "system", "content": "You are a smart data‑analysis agent."},
        {"role": "user", "content": task},
    ]

    resp = await openai.ChatCompletion.acreate(
        model=GPT_MODEL,
        messages=messages,
        tools=tools_schema,
        tool_choice="auto",
        stream=False,
    )

    async def dispatch(call):
        fn_name = call["function_call"]["name"]
        args = json.loads(call["function_call"]["arguments"])
        if fn_name == "scrape_website":
            return await scrape_website(**args)
        if fn_name == "get_relevant_data":
            return await get_relevant_data(**args)
        if fn_name == "answer_questions":
            return await answer_questions(**args)
        raise ValueError(f"Unknown tool {fn_name}")

    # Iterate if model returns multiple tool steps (simple linear loop)
    while resp.choices[0].finish_reason == "tool_calls":
        results = []
        for call in resp.choices[0].message.tool_calls:
            results.append(await dispatch(call))
        messages.append(resp.choices[0].message)
        messages.append({"role": "tool", "content": json.dumps(results)})
        resp = await openai.ChatCompletion.acreate(
            model=GPT_MODEL,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
            stream=False,
        )

    final_json = json.loads(resp.choices[0].message.content)
    return AnswerPayload(answers=final_json)
