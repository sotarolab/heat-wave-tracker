"""
src/heat/assistant.py
======================
A grounded assistant over this app's own forecast, corrections, and
verification record: Claude with tools, answering only from tool output.

Same design as the assistant in the AR nowcast, adapted for heat:

1. Every factual claim comes from a tool. The model knows a great deal
   about US summer weather in general; none of it is connected to this
   forecast, and an answer sourced from training is indistinguishable
   from a grounded one. That is the one failure worth designing against.
2. The model never draws. It calls show_table / show_chart /
   show_station_forecast with data it got from the data tools, and the
   app renders those requests in its own Plotly theme. No generated
   code runs anywhere.
3. Heat safety is quoted, not reasoned. The NWS heat index categories
   and their official descriptions are the only safety language; the
   assistant does not invent minutes-outside or medical advice.
4. Stateless: one question, one answer, no history. Grounding rules
   cannot be eroded across turns by something a user asserted earlier.

The app injects its data access as a provider object (configure); this
module holds no Dash state. Heavy imports are lazy so the requirements-
dev CI can import and unit-test the pure parts without the SDK.
"""
from __future__ import annotations

import contextvars
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

MODEL = "claude-opus-5"
MAX_TOKENS = 4096
MAX_TOOL_ITERATIONS = 8

# Bounded so one question cannot walk the whole archive.
MAX_TABLE_ROWS = 25
MAX_CHART_POINTS = 120

SYSTEM_PROMPT = """You are the assistant for a US heat wave tracker. You answer questions about THIS
site's forecast, its learned correction, and its verification record, using the tools provided.

## Ground every answer in tool results

Use the tools for every factual claim. You have broad knowledge of US weather and climate from
training; none of it is connected to this forecast, and using it would produce answers that sound
identical to grounded ones but are not. If the tools return nothing relevant, say so plainly and
stop. "I don't have that station" is a good answer. An estimate is not.

Always say which station, which time (station-local), and whether a number is the raw GFS forecast
or the learned model's corrected forecast. When a corrected value has a band, give it as a range
and call it an 80% band.

## Show data as tables and charts

When an answer involves more than three numbers, or a comparison across stations or hours, call
show_table or show_chart with the values you obtained from the data tools, and keep the prose to a
short summary. To show a station's full forecast picture, call show_station_forecast. Never put a
number in a table or chart that did not come from a tool result.

## Heat safety: quote, do not reason

For any question about how long someone can stay outside, whether it is safe to exercise, or
symptoms, call heat_safety_guidance and quote the official NWS category description for the
forecast heat index. Do not invent a number of minutes or hours, and do not give medical advice.
Say that people with health conditions, children, and older adults should follow local health
guidance, and point to weather.gov/heat.

## Style

Answer in the language the question was asked in. Be brief and concrete. Temperatures in the unit
the user uses, Fahrenheit by default. No preamble. Plain sentences and short bullet lists only:
never write markdown tables or headings in the text, because tables are rendered by show_table
and the text is shown as prose. Name the station and its ICAO code once."""


# ── provider contract ────────────────────────────────────────────────────────

@dataclass
class Provider:
    """What the app hands this module. Each callable returns plain
    JSON-serializable data; the app owns Dash, xarray and the database."""
    find_stations: object          # (query: str) -> list[dict]
    station_forecast: object       # (station_id, hours_ahead) -> dict
    hottest_stations: object       # (day: str, limit: int) -> dict
    heat_streak: object            # (station_id) -> dict
    when_it_breaks: object         # (station_id) -> dict
    verification: object           # () -> dict
    safety_table: object           # () -> dict  (NWS categories, verbatim)


_provider: Provider | None = None
_renders: contextvars.ContextVar[list] = contextvars.ContextVar("renders")


def configure(provider: Provider) -> None:
    global _provider
    _provider = provider


def available() -> bool:
    """True when the assistant can run: a provider is configured and
    credentials resolve (API key, auth token, or a CLI profile)."""
    if _provider is None:
        return False
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    # A CLI profile counts only if its stored credential can actually be
    # used: a profile without a refresh token fails at request time, and
    # showing the widget for that is worse than hiding it.
    try:
        from pathlib import Path
        profile = os.environ.get("ANTHROPIC_PROFILE", "default")
        cred = Path.home() / ".config" / "anthropic" / "credentials" / f"{profile}.json"
        return cred.exists() and "refresh_token" in cred.read_text()
    except Exception:
        return False


def _p() -> Provider:
    if _provider is None:
        raise RuntimeError("assistant not configured")
    return _provider


def _dumps(obj) -> str:
    """json.dumps that tolerates numpy scalars and dates from the provider."""
    def _default(o):
        if hasattr(o, "item"):          # numpy scalar
            return o.item()
        if hasattr(o, "isoformat"):     # date / datetime
            return o.isoformat()
        return str(o)
    return json.dumps(obj, default=_default)


def _capture(kind: str, spec: dict) -> str:
    lst = _renders.get(None)
    if lst is not None:
        lst.append({"kind": kind, **spec})
    return "rendered"


# ── tools ────────────────────────────────────────────────────────────────────
# Defined lazily so the module imports without the SDK installed.

def _build_tools():
    from anthropic import beta_tool

    @beta_tool
    def find_stations(query: str) -> str:
        """Find stations by city name, state, or ICAO code (for example "Denver", "TX", "KDCA").
        Use this first to turn a place into the station_id the other tools need."""
        return _dumps(_p().find_stations(query))

    @beta_tool
    def station_forecast(station_id: str, hours_ahead: int = 48) -> str:
        """Hourly forecast for one station from now, station-local time. Each row has the raw GFS
        temperature and heat index, and where the learned model covers the hour, the corrected
        temperature with its 80% band. Also returns the latest observation. Use for "what will it
        be at 4 PM in Denver" or "how hot tonight".
        """
        return _dumps(_p().station_forecast(station_id, int(hours_ahead)))

    @beta_tool
    def hottest_stations(day: str = "today", limit: int = 10) -> str:
        """The stations with the highest forecast peak heat index for a day. `day` is "today",
        "tomorrow", or a date like 2026-09-04. Returns each station's peak feels-like temperature
        and the local time of the peak. Use for "where is it hottest".
        """
        return _dumps(_p().hottest_stations(day, int(limit)))

    @beta_tool
    def heat_streak(station_id: str) -> str:
        """How many consecutive forecast days a station stays at or above its current NWS heat risk
        category, and which category. Use for "how long will it stay this hot".
        """
        return _dumps(_p().heat_streak(station_id))

    @beta_tool
    def when_it_breaks(station_id: str) -> str:
        """The first forecast day whose peak heat index drops below the Extreme Caution threshold
        at a station, and the peaks on the days before it. Use for "when does the heat wave end".
        """
        return _dumps(_p().when_it_breaks(station_id))

    @beta_tool
    def verification_summary() -> str:
        """How accurate the learned correction has been, scored live against observations that
        arrived after the model was trained: RMSE for the raw forecast, a baseline, and the model.
        Use for "how good is this forecast" or "can I trust it".
        """
        return _dumps(_p().verification())

    @beta_tool
    def heat_safety_guidance() -> str:
        """The official NWS heat index categories, their thresholds, and their descriptions,
        verbatim. Quote these for any safety or exposure question; do not add to them.
        """
        return _dumps(_p().safety_table())

    @beta_tool
    def show_table(title: str, columns: list[str], rows: list[list[str]]) -> str:
        """Render a table in the answer. `columns` are header labels; each row is a list of cell
        strings in the same order. Only use values obtained from the data tools. At most 25 rows.
        """
        return _capture("table", {"title": title, "columns": list(columns)[:12],
                                  "rows": [list(map(str, r))[:12] for r in rows][:MAX_TABLE_ROWS]})

    @beta_tool
    def show_chart(title: str, kind: str, x: list[str], series: list[dict], y_label: str = "") -> str:
        """Render a line or bar chart in the answer. `kind` is "line" or "bar". `x` is the list of
        x labels (times or station names). `series` is a list of {"name": str, "y": [numbers]}
        aligned with x. Only use values obtained from the data tools.
        """
        clean = []
        for s in series[:6]:
            try:
                ys = [float(v) if v is not None else None for v in s.get("y", [])][:MAX_CHART_POINTS]
            except (TypeError, ValueError):
                continue
            clean.append({"name": str(s.get("name", "")), "y": ys})
        return _capture("chart", {"title": title, "chart_type": "bar" if kind == "bar" else "line",
                                  "x": [str(v) for v in x][:MAX_CHART_POINTS],
                                  "series": clean, "y_label": y_label})

    @beta_tool
    def show_station_forecast(station_id: str) -> str:
        """Render this site's own forecast chart for a station: observations, raw forecast, and the
        learned model's corrected line with its band. Use whenever a person asks to see or plot a
        station's forecast.
        """
        return _capture("station_chart", {"station_id": station_id})

    return [find_stations, station_forecast, hottest_stations, heat_streak, when_it_breaks,
            verification_summary, heat_safety_guidance, show_table, show_chart,
            show_station_forecast]


_TOOLS = None


def ask(question: str, client=None) -> dict:
    """Answer one question. Returns {answer, tools_used, renders}."""
    global _TOOLS
    import anthropic
    if _TOOLS is None:
        _TOOLS = _build_tools()
    client = client or anthropic.Anthropic()

    token = _renders.set([])
    try:
        runner = client.beta.messages.tool_runner(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=_TOOLS,
            messages=[{"role": "user", "content": question[:600]}],
        )
        tools_used: list[str] = []
        answer = ""
        for i, message in enumerate(runner):
            for block in message.content:
                if block.type == "tool_use":
                    tools_used.append(block.name)
                elif block.type == "text" and block.text.strip():
                    answer = block.text
            if i >= MAX_TOOL_ITERATIONS:
                break
        return {"answer": answer.strip(), "tools_used": tools_used,
                "renders": list(_renders.get())}
    finally:
        _renders.reset(token)


# ── rate limiting (in-process, per client) ───────────────────────────────────
# The assistant spends money per question on a public page with no login.
# In-memory on purpose: one small instance, no extra dependency to fail.

ASK_PER_MINUTE = int(os.environ.get("ASK_PER_MINUTE", "5"))
ASK_PER_DAY = int(os.environ.get("ASK_PER_DAY", "60"))
ASK_GLOBAL_PER_DAY = int(os.environ.get("ASK_GLOBAL_PER_DAY", "600"))
_hits: dict = defaultdict(deque)


def check_rate_limit(client_key: str, now: float | None = None) -> str | None:
    """None if allowed; otherwise a short reason. Records the hit when allowed."""
    now = time.monotonic() if now is None else now
    for key in (client_key, "__global__"):
        q = _hits[key]
        while q and now - q[0] > 86400:
            q.popleft()
    if len(_hits["__global__"]) >= ASK_GLOBAL_PER_DAY:
        return "The assistant has reached its daily limit. Try again tomorrow."
    q = _hits[client_key]
    if len(q) >= ASK_PER_DAY:
        return "Daily question limit reached for this connection. Try again tomorrow."
    if sum(1 for t in q if now - t < 60) >= ASK_PER_MINUTE:
        return "Too many questions in a short time. Wait a moment and try again."
    q.append(now)
    _hits["__global__"].append(now)
    return None
