# AutoStream — Social-to-Lead Agentic Workflow

A conversational AI agent built for **AutoStream**, a fictional SaaS video editing platform.
The agent handles product queries, detects high-intent users, and captures leads — all within
a stateful multi-turn conversation powered by **LangGraph** + **Claude Haiku**.

---

## Project Structure

```
autostream_agent/
├── agent/
│   ├── __init__.py
│   ├── graph.py          # LangGraph state machine + all nodes
│   ├── rag.py            # Knowledge retrieval from local JSON
│   └── tools.py          # mock_lead_capture + lead field extraction
├── data/
│   └── knowledge_base.json   # Pricing, features, policies
├── main.py               # CLI entry point
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/autostream-agent.git
cd autostream-agent
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your API key

This project uses **Claude Haiku** via the Anthropic API.

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

Get a free API key at [console.anthropic.com](https://console.anthropic.com).

> To use GPT-4o-mini or Gemini instead, replace `ChatAnthropic` in `agent/graph.py`
> with `ChatOpenAI` or `ChatGoogleGenerativeAI` respectively.

### 5. Run the agent
```bash
python main.py
```

---

## Example Conversation

```
You: Hi there!
AutoStream: Hey! Welcome to AutoStream 👋 I'm here to help you find the right plan...

You: What's included in the Pro plan?
AutoStream: The Pro plan is $79/month and includes unlimited videos, 4K resolution,
AI captions, and 24/7 priority support...

You: That sounds great, I want to sign up for the YouTube channel I run.
AutoStream: Awesome, I'd love to get you set up! 🚀  Could I start with your name?

You: Alex Johnson
AutoStream: Great, Alex! What's the best email address to reach you at?

You: alex@example.com
AutoStream: Almost there! Which platform do you primarily create for?

You: YouTube
AutoStream: Perfect, thank you Alex! 🎉  I've passed your details to our team...

[TOOL CALLED] mock_lead_capture('Alex Johnson', 'alex@example.com', 'YouTube')
```

---

## Architecture 

### Why LangGraph?

LangGraph was chosen over AutoGen because this use case is a **linear, stateful pipeline**
with well-defined routing logic — not a multi-agent debate or code execution loop.
LangGraph's `StateGraph` gives explicit control over which node runs next, making the
intent → response → lead-capture flow easy to reason about and debug.

### How state is managed

All conversation state lives in `AgentState`, a typed dictionary passed through every graph node:

- **`messages`** — the full conversation history (using LangGraph's `add_messages` reducer,
  which appends rather than replaces). This gives the LLM memory across all turns.
- **`intent`** — the latest classified intent (`greeting`, `product_query`, `high_intent`),
  written by the `intent_router` node and read by the conditional edge to decide routing.
- **`lead`** — a `LeadData` dataclass tracking how many lead fields (name, email, platform)
  have been collected. The `lead_collector` node reads from and writes back to this field
  each turn until all three are filled.
- **`lead_captured`** — boolean flag that prevents the mock tool from being called twice.

### Flow

```
START → intent_router → [greeter | rag_responder | lead_collector] → END
```

The conditional edge reads `state["intent"]` (and whether lead collection is in progress)
to choose the next node. The LLM always receives the full message history, ensuring
coherent multi-turn reasoning.

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp:

1. **Register a WhatsApp Business API account** via Meta for Developers and create an app
   with the WhatsApp product enabled.

2. **Set up a webhook endpoint** (e.g. using FastAPI or Flask):
   ```python
   @app.post("/webhook")
   async def webhook(request: Request):
       body = await request.json()
       message = body["entry"][0]["changes"][0]["value"]["messages"][0]
       user_id = message["from"]         # WhatsApp phone number
       text = message["text"]["body"]    # User's message
       reply = await run_agent(user_id, text)
       send_whatsapp_message(user_id, reply)
   ```

3. **Persist state per user** — replace the in-memory `AgentState` dict with a database
   (Redis or PostgreSQL), keyed by `user_id`. On each webhook call, load that user's state,
   run the graph, and save the updated state back.

4. **Send the reply** using the WhatsApp Cloud API:
   ```python
   requests.post(
       f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages",
       headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
       json={"messaging_product": "whatsapp", "to": user_id,
             "type": "text", "text": {"body": reply}}
   )
   ```

5. **Verify the webhook** by handling the `GET /webhook` request Meta sends during setup,
   checking the `hub.verify_token` against your secret.

The core agent logic requires no changes — only the state persistence and I/O layer
needs to be adapted for the async webhook model.
