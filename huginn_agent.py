#!/usr/bin/env python3
"""
huginn_agent.py
---------------
A local agentic AI tool that:

  1. Monitors a text file for changes (watchdog)
  2. Sends each new task to a local Ollama LLM (qwen2.5:7b) via langchain-ollama
  3. Equips the agent with a DuckDuckGo web-search tool
  4. Runs a LangGraph ReAct loop (Thought → Action → Observation) until a
     Final Answer is produced
  5. Emails the answer via a configurable SMTP server (smtplib)

Configuration — set these in a .env file or in your shell environment:

  TASKS_FILE        Path to the watched file      (default: ./tasks.txt)
  OLLAMA_BASE_URL   Ollama API endpoint            (default: http://localhost:11434)
  SMTP_HOST         SMTP server hostname           (required to send email)
  SMTP_PORT         SMTP server port               (default: 587)
  SMTP_USER         SMTP login username
  SMTP_PASSWORD     SMTP login password
  EMAIL_SENDER      From address                   (defaults to SMTP_USER)
  EMAIL_RECIPIENT   Destination address

If SMTP_HOST is empty the agent still runs but prints the Final Answer to
the console instead of sending an email.
"""

import os
import time
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.prebuilt import create_react_agent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load environment variables from a .env file if one exists
# ---------------------------------------------------------------------------
load_dotenv()

# --- file to watch ---
TASKS_FILE = Path(os.getenv("TASKS_FILE", "./tasks.txt")).expanduser().resolve()

# --- Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- email / SMTP ---
SMTP_HOST      = os.getenv("SMTP_HOST", "")
SMTP_PORT      = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER      = os.getenv("SMTP_USER", "")
SMTP_PASSWORD  = os.getenv("SMTP_PASSWORD", "")
EMAIL_SENDER   = os.getenv("EMAIL_SENDER", SMTP_USER)   # fall back to login name
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", "")


# ---------------------------------------------------------------------------
# Email helper
# ---------------------------------------------------------------------------
def send_email(subject: str, body: str) -> None:
    """
    Send a plain-text email via the configured SMTP server using STARTTLS.

    If SMTP_HOST is not configured the function falls back to printing
    the answer to stdout so the agent still works without email setup.
    """
    if not SMTP_HOST:
        log.warning(
            "SMTP_HOST is not set – skipping email and printing the answer instead."
        )
        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(body)
        print("=" * 60 + "\n")
        return

    # Build a multipart message so mail clients render it cleanly
    msg = MIMEMultipart()
    msg["From"]    = EMAIL_SENDER
    msg["To"]      = EMAIL_RECIPIENT
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        # Use a context manager so the connection is always closed cleanly
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()               # identify ourselves to the server
            server.starttls()           # upgrade the socket to TLS
            server.ehlo()               # re-identify over the encrypted channel
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(EMAIL_SENDER, [EMAIL_RECIPIENT], msg.as_string())
        log.info("Email sent successfully to %s", EMAIL_RECIPIENT)
    except smtplib.SMTPException as exc:
        log.error("SMTP error while sending email: %s", exc)
    except OSError as exc:
        log.error("Network error while sending email: %s", exc)


# ---------------------------------------------------------------------------
# LLM + LangGraph ReAct agent  (built once at startup)
# ---------------------------------------------------------------------------
def build_agent():
    """
    Initialise the ChatOllama model and DuckDuckGo search tool, then compile
    a LangGraph ReAct agent that owns both.

    The ReAct pattern works like this:
      Thought  → the LLM reasons about what to do next
      Action   → the LLM calls a tool (e.g. web_search)
      Observation → the tool result is fed back to the LLM
      … repeat until the LLM emits a final answer without a tool call …
    """
    log.info(
        "Initialising ChatOllama (model=qwen2.5:7b, base_url=%s) …", OLLAMA_BASE_URL
    )
    llm = ChatOllama(
        model="qwen2.5:7b",
        base_url=OLLAMA_BASE_URL,
        temperature=0,          # deterministic; good for research tasks
    )

    # DuckDuckGo search — no API key required
    search_tool = DuckDuckGoSearchResults(
        name="web_search",
        description=(
            "Search the internet for up-to-date information. "
            "Use this whenever you need facts, news, or data you are not sure about. "
            "Input: a plain-text search query string."
        ),
        num_results=5,          # number of snippets returned per query
    )

    # create_react_agent compiles a LangGraph StateGraph that repeatedly
    # invokes the LLM and any tools until the model stops issuing tool calls.
    agent = create_react_agent(model=llm, tools=[search_tool])
    log.info("Agent ready.")
    return agent


# Build the agent once so we reuse the same LLM connection for every task
AGENT = build_agent()


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------
def run_task(task_text: str) -> None:
    """
    Feed *task_text* to the ReAct agent, wait for the agentic loop to finish,
    then email (or print) the final answer.
    """
    task_text = task_text.strip()
    if not task_text:
        log.info("tasks.txt is empty – nothing to do.")
        return

    log.info("Starting agent for task:\n  %s", task_text[:200])

    try:
        # LangGraph agents expect a dict with a "messages" key.
        # Each entry can be a (role, content) tuple or a BaseMessage object.
        result = AGENT.invoke(
            {"messages": [("user", task_text)]},
            # recursion_limit caps the maximum number of ReAct iterations to
            # prevent infinite loops on pathological inputs.
            config={"recursion_limit": 30},
        )
    except Exception as exc:
        log.error("Agent raised an unexpected error: %s", exc, exc_info=True)
        return

    # The final message in the graph output is the model's last response —
    # this is the "Final Answer" once the model stops calling tools.
    final_message = result["messages"][-1]
    final_answer = (
        final_message.content
        if hasattr(final_message, "content")
        else str(final_message)
    )

    log.info("Agent finished. Dispatching answer …")

    # Truncate the subject line to keep it readable in email clients
    short_task = task_text[:60] + ("…" if len(task_text) > 60 else "")
    send_email(
        subject=f"[Huginn] {short_task}",
        body=final_answer,
    )


# ---------------------------------------------------------------------------
# Watchdog file-system event handler
# ---------------------------------------------------------------------------
class TaskFileHandler(FileSystemEventHandler):
    """
    Reacts to file-modification events on TASKS_FILE.

    Watchdog monitors the *directory* that contains the file, so we must
    filter events down to the specific file we care about.

    A 1-second debounce prevents double-firing that some editors cause by
    writing the file in multiple flush operations (e.g. atomic saves that
    first write to a temp file then rename).
    """

    def __init__(self) -> None:
        super().__init__()
        self._last_modified: float = 0.0    # monotonic timestamp of last handled event

    def on_modified(self, event) -> None:
        # Skip directory-change events and events for other files
        if event.is_directory:
            return
        if Path(event.src_path).resolve() != TASKS_FILE:
            return

        # --- debounce ---
        now = time.monotonic()
        if now - self._last_modified < 1.0:
            return
        self._last_modified = now

        log.info("Change detected in %s", TASKS_FILE)

        try:
            content = TASKS_FILE.read_text(encoding="utf-8")
        except OSError as exc:
            log.error("Could not read %s: %s", TASKS_FILE, exc)
            return

        # Hand off to the task runner (blocking call — the observer thread
        # will queue further events while this runs, which is fine)
        run_task(content)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    # Create the tasks file if it does not exist yet so users can find it
    TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not TASKS_FILE.exists():
        TASKS_FILE.write_text("", encoding="utf-8")
        log.info("Created empty task file at %s", TASKS_FILE)

    log.info("Watching %s for changes …  (Ctrl-C to stop)", TASKS_FILE)

    event_handler = TaskFileHandler()
    observer = Observer()

    # watchdog requires a directory path, not a file path
    observer.schedule(event_handler, str(TASKS_FILE.parent), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)           # main thread just keeps the process alive
    except KeyboardInterrupt:
        log.info("Interrupt received. Shutting down …")
    finally:
        observer.stop()
        observer.join()
        log.info("Goodbye.")


if __name__ == "__main__":
    main()
