import subprocess, json, textwrap, tempfile, uuid, base64, io
from pathlib import Path
from typing import Any, Dict
import requests
from bs4 import BeautifulSoup

SCRAPE_TIMEOUT = 30

# ---------------------------------------------------------------------------
async def scrape_website(url: str) -> str:
    """Fetch raw HTML using requests (lighter than Playwright)."""
    resp = requests.get(url, timeout=SCRAPE_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    html_path = Path(tempfile.gettempdir()) / f"scraped_{uuid.uuid4().hex}.html"
    html_path.write_text(resp.text, encoding="utf-8")
    return str(html_path)

# ---------------------------------------------------------------------------
async def get_relevant_data(html_file: str, selector: str) -> str:
    """Extract table/list/paragraphs matching a CSS selector, return JSON."""
    soup = BeautifulSoup(Path(html_file).read_text(encoding="utf-8"), "html.parser")
    nodes = soup.select(selector)
    data = [n.get_text(" ", strip=True) for n in nodes]
    json_path = Path(tempfile.gettempdir()) / f"slice_{uuid.uuid4().hex}.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return str(json_path)

# ---------------------------------------------------------------------------
async def answer_questions(python_code: str) -> str:
    """Run LLM‑generated code in an isolated subprocess and capture stdout."""
    safe_code = textwrap.dedent(python_code)
    script_path = Path(tempfile.gettempdir()) / f"exec_{uuid.uuid4().hex}.py"
    script_path.write_text(safe_code, encoding="utf-8")
    proc = subprocess.run([
        "python", "-I", "-S", "-B", "-E", str(script_path)
    ], capture_output=True, text=True, timeout=120)
    proc.check_returncode()
    return proc.stdout.strip()
