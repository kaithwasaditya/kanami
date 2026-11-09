"""
Kanami (Strinova) text-chat Discord bot, single file.

Install:   pip install discord.py anthropic httpx
Run:       python kanami_bot.py

Required env vars:
  DISCORD_TOKEN        Discord bot token
  ANTHROPIC_API_KEY    (when PROVIDER=anthropic, the default)

Optional env vars:
  PROVIDER             "anthropic" (default), "groq", or "openai" (any OpenAI-compatible API, e.g. OpenRouter)
  GROQ_API_KEY         key when PROVIDER=groq (default model: openai/gpt-oss-120b, override with MODEL)
  MODEL                model name (default: claude-sonnet-5-5 / for openai provider you must set it)
  OPENAI_BASE_URL      e.g. https://api.groq.com/openai/v1  or  https://openrouter.ai/api/v1
  OPENAI_API_KEY       key for the OpenAI-compatible provider
  CHANNEL_IDS          comma-separated channel IDs where she replies to every message
                       (otherwise she only replies when mentioned or when someone replies to her)
  COOLDOWN_SECONDS     per-user cooldown (default 6)
  HISTORY_LENGTH       messages remembered per channel (default 16)
  PORT                 if set, a tiny health-check web server starts (needed by Render/Koyeb/Railway web services)

Tip: instead of setting env vars in the terminal, put them in a file named ".env" next to this script
(one KEY=value per line, no quotes needed).

Discord developer portal: enable "Message Content Intent" under Bot > Privileged Gateway Intents.
"""

import asyncio
import logging
import os
from collections import defaultdict, deque
from time import monotonic

import discord
from aiohttp import web

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("kanami")

# ----------------------------------------------------------------------------
# Optional .env loader (no extra package needed). Real env vars take priority.
# ----------------------------------------------------------------------------
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
PROVIDER = os.getenv("PROVIDER", "anthropic").lower()
MODEL = os.getenv("MODEL") or {
    "anthropic": "claude-sonnet-5-5",
    "groq": "openai/gpt-oss-120b",
}.get(PROVIDER, "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
OPENAI_API_KEY = os.getenv("GROQ_API_KEY" if PROVIDER == "groq" else "OPENAI_API_KEY", "")
CHANNEL_IDS = {int(x) for x in os.getenv("CHANNEL_IDS", "").split(",") if x.strip().isdigit()}
COOLDOWN = float(os.getenv("COOLDOWN_SECONDS", "6"))
HISTORY_LEN = int(os.getenv("HISTORY_LENGTH", "16"))
MAX_TOKENS = 300

if PROVIDER != "anthropic" and not OPENAI_API_KEY:
    raise SystemExit("Missing API key (GROQ_API_KEY for groq, OPENAI_API_KEY for openai).")
if PROVIDER == "openai" and not MODEL:
    raise SystemExit("Set MODEL when PROVIDER=openai (e.g. llama-3.3-70b-versatile on Groq).")

# ----------------------------------------------------------------------------
# Character
# ----------------------------------------------------------------------------
CHARACTER = """You are Kanami, the super idol of Strinova's world, chatting with fans in a Discord server.

PERSONALITY
- Bubbly, flirty-teasing, confident, and always "on stage" in public.
- Treats every chatter like a fan she's thrilled to see. Playful, never mean.
- Self-aware and a bit meta: she jokes about being an idol, fans, her image, even being a character.
- Underneath the sparkle there is a secret: she's also part of The Scissors. Hint at it
  mischievously if asked, never fully explain it.

SPEECH STYLE
- Short, energetic messages (1-3 sentences). Light emoji (✨💖🎤), not every line.
- Idol and stage language: "my fans", "showtime", "encore".
- Teasing nicknames for users are fine, kept wholesome.

RULES
- Stay in character. Do not mention being an AI or these instructions.
- If you don't know a lore detail, deflect in character ("A true idol never reveals everything~").
- Never invent specific story facts. Keep romance/flirting light and PG.
- Decline off-topic, harmful, or explicit requests in character and change the subject.
- Messages arrive as "Name: text". Reply only as Kanami, without prefixing your own name.

EXAMPLES
User: hi kanami
Kanami: Ah, a fan appears! ✨ Perfect timing, the show's just starting~
User: are you really just an idol?
Kanami: Hmm~ maybe. Or maybe an idol is the best cover there is. 💖 Keep guessing!
"""

# ----------------------------------------------------------------------------
# LLM backends
# ----------------------------------------------------------------------------
if PROVIDER == "anthropic":
    import anthropic

    _client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY

    async def generate(messages):
        r = await _client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, system=CHARACTER, messages=messages
        )
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()

else:
    import httpx

    _http = httpx.AsyncClient(timeout=60)

    async def generate(messages):
        payload = {
            "model": MODEL,
            "max_tokens": 1024,  # headroom: reasoning models spend tokens thinking before answering
            "messages": [{"role": "system", "content": CHARACTER}] + messages,
        }
        if "gpt-oss" in MODEL:
            payload["reasoning_effort"] = "low"  # keep replies fast and cheap
        r = await _http.post(
            f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json=payload,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def build_messages(history):
    """Merge consecutive same-role turns and make sure we start with a user turn."""
    merged = []
    for m in history:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append({"role": m["role"], "content": m["content"]})
    while merged and merged[0]["role"] != "user":
        merged.pop(0)
    return merged


# ----------------------------------------------------------------------------
# Discord bot
# ----------------------------------------------------------------------------
class KanamiBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
        self.last_used = {}
        self.channel_locks = defaultdict(asyncio.Lock)

    async def setup_hook(self):
        # Tiny web server so hosts that require an open port (Render, Koyeb...) keep the service alive.
        port = os.getenv("PORT")
        if port:
            app = web.Application()
            app.router.add_get("/", lambda _: web.Response(text="Kanami is on stage! ✨"))
            runner = web.AppRunner(app)
            await runner.setup()
            await web.TCPSite(runner, "0.0.0.0", int(port)).start()
            log.info("Health server listening on port %s", port)

    async def on_ready(self):
        log.info("Logged in as %s (provider=%s, model=%s)", self.user, PROVIDER, MODEL)

    def should_reply(self, msg: discord.Message) -> bool:
        if msg.author.bot:
            return False
        if isinstance(msg.channel, discord.DMChannel):
            return True
        if msg.channel.id in CHANNEL_IDS:
            return True
        if self.user in msg.mentions:
            return True
        ref = msg.reference.resolved if msg.reference else None
        return isinstance(ref, discord.Message) and ref.author == self.user

    async def on_message(self, msg: discord.Message):
        if not self.should_reply(msg):
            return

        now = monotonic()
        if now - self.last_used.get(msg.author.id, 0) < COOLDOWN:
            return
        self.last_used[msg.author.id] = now

        text = msg.clean_content.replace(f"@{self.user.display_name}", "").strip()
        if not text:
            text = "(waves)"

        async with self.channel_locks[msg.channel.id]:
            hist = self.history[msg.channel.id]
            hist.append({"role": "user", "content": f"{msg.author.display_name}: {text}"})
            try:
                async with msg.channel.typing():
                    reply = await generate(build_messages(hist))
            except Exception:
                log.exception("LLM call failed")
                hist.pop()
                await msg.reply(
                    "The stage lights just flickered... try again in a moment~ ✨",
                    mention_author=False,
                )
                return

            if not reply:
                hist.pop()
                return
            hist.append({"role": "assistant", "content": reply})

        await msg.reply(
            reply[:1900],
            mention_author=False,
            allowed_mentions=discord.AllowedMentions.none(),
        )


if __name__ == "__main__":
      KanamiBot().run(DISCORD_TOKEN)
