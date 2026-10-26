# AI Travel Planner

A small AI travel planning app built for the Express Analytics AI/ML Engineer take-home assignment. It uses FastAPI for the API, LangGraph to manage the planning workflow, Groq for LLM calls, Serper for web search, Open-Meteo for weather data, and SQLite to save workflow state.

## What it does

- Takes a destination, dates, budget, interests, traveller count, and preferences.
- Researches attractions, recent local information, events, and weather.
- Generates a day-by-day itinerary with estimated costs.
- Checks dates, budget arithmetic, activity sources, and research-related issues.
- Pauses for a person to approve the draft, reject it with feedback, or request modifications.
- Saves graph state in SQLite so a paused plan can be resumed.
- Provides a basic web form at `/` as well as a REST API.

Detailed itineraries are limited to 14 calendar days per request. Past start dates and invalid date ranges are rejected before research begins.

## Architecture

```text
Browser / API client
        |
        v
     FastAPI
        |
        v
 Request validation
        |
        v
 Research Agent ------ Serper web search and news
        |               Open-Meteo weather and geocoding
        v
 Planner Agent ------- Activity Lookup tool
        |               Budget Calculator tool
        v
 Validator Agent ----- Deterministic checks + advisory LLM review
        |
        v
 Human review (LangGraph interrupt)
        | approve              | reject / modify
        v                      v
 Final itinerary        Planner -> Validator -> Review
        |
        v
 SQLite checkpoint storage
```

### Main components

- **Research Agent** (`app/agents/researcher.py`): gathers destination evidence using Serper and weather data using Open-Meteo.
- **Planner Agent** (`app/agents/planner.py`): uses Groq to produce a structured itinerary from the request and researched candidates.
- **Activity Lookup** (`app/tools/activities.py`): filters and ranks sourced attractions and upcoming events.
- **Budget Calculator** (`app/tools/budget.py`): calculates a consistent budget allocation.
- **Validator Agent** (`app/agents/itinerary_validator.py`): checks dates, costs, and source grounding, then adds advisory quality findings.
- **LangGraph workflow** (`app/graph/`): manages the steps, human-review pause, revisions, and final approval.
- **SQLite checkpointer**: persists workflow state across the review pause.

The validator does not approve or rewrite a plan. The user makes the final decision. Cost values are estimates, not live hotel, flight, or ticket prices. The app does not have a live routing engine, so travel times and schedule feasibility may need manual confirmation.

## Requirements

- Python 3.11 or newer
- A [Serper API key](https://serper.dev)
- A [Groq API key](https://console.groq.com)

Open-Meteo's public geocoding and weather endpoints do not require an API key.

## Run locally

```bash
git clone https://github.com/Farhan-Pagarkar/ai-travel-planner.git
cd ai-travel-planner
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your keys to `.env`:

```env
SERPER_API_KEY=your_serper_api_key
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
CHECKPOINT_DB_PATH=data/checkpoints.sqlite
```

Start the application:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open these URLs:

- Web interface: <http://127.0.0.1:8000/>
- API documentation: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/plan` | Create a plan and pause at human review |
| `GET` | `/plan/{plan_id}` | Get the current plan and workflow status |
| `POST` | `/plan/{plan_id}/review` | Approve, reject, or modify a draft |
| `GET` | `/plan/{plan_id}/final` | Get the itinerary after approval |

### Create a plan

```bash
curl -X POST http://127.0.0.1:8000/plan \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "Kyoto, Japan",
    "start_date": "2026-11-01",
    "end_date": "2026-11-04",
    "total_trip_budget": 2500,
    "currency": "USD",
    "interests": ["temples", "food", "gardens"],
    "traveler_count": 2,
    "preferences": {"pace": "moderate"}
  }'
```

The response includes a `plan_id`, an `awaiting_review` status, the draft itinerary, research results, and any validation findings. Use the returned plan ID in later requests.

### Approve a plan

```bash
curl -X POST http://127.0.0.1:8000/plan/$PLAN_ID/review \
  -H "Content-Type: application/json" \
  -d '{"action": "approve"}'
```

### Request a revision

Reject with feedback:

```bash
curl -X POST http://127.0.0.1:8000/plan/$PLAN_ID/review \
  -H "Content-Type: application/json" \
  -d '{"action": "reject", "feedback": "Keep each day in one area and add more cultural activities."}'
```

Or request specific changes:

```bash
curl -X POST http://127.0.0.1:8000/plan/$PLAN_ID/review \
  -H "Content-Type: application/json" \
  -d '{"action": "modify", "modifications": "Replace the second day's afternoon activity with a garden visit."}'
```

After approval, retrieve the final itinerary:

```bash
curl http://127.0.0.1:8000/plan/$PLAN_ID/final
```

The final endpoint returns a conflict if the plan has not been approved. The workflow allows up to three revision cycles.

## Design choices and limitations

- **LangGraph** manages the workflow and review pause rather than custom polling logic.
- **SQLite** keeps checkpointing simple for local use and a single application instance.
- **Deterministic checks** handle hard rules such as date coverage, cost sums, and source matching. The LLM review adds advice but is not a substitute for human approval.
- **Research coverage varies** by destination and search results. If few suitable candidates are found, the itinerary may contain generic activities and warnings.
- **Weather forecasts** are limited to the available forecast horizon; dates farther out use historical weather comparisons, not a forecast.
- **Costs and logistics** are estimates. There are no live hotel or flight bookings, real-time prices, or routing calculations.

With more time, I would add a routing API, better cost sources, and more integration tests against real destination examples. For production, I would consider PostgreSQL for checkpoint storage, request timeouts and rate limits, monitoring, and stronger handling of provider outages.
