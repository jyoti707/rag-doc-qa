"""One small wrapper for all LLM providers. Gemini, Groq, Ollama and OpenAI share the OpenAI-compatible API."""
import os, time

PROVIDERS = {
    "extractive (no LLM, offline)": dict(kind="none", model="", base_url=None, env=None),
    "openai":    dict(kind="openai", model="gpt-4o-mini", base_url=None, env="OPENAI_API_KEY"),
    "groq":      dict(kind="openai", model="llama-3.3-70b-versatile", base_url="https://api.groq.com/openai/v1", env="GROQ_API_KEY"),
    "gemini":    dict(kind="openai", model="gemini-2.5-flash", base_url="https://generativelanguage.googleapis.com/v1beta/openai/", env="GEMINI_API_KEY"),
    "ollama (local)": dict(kind="openai", model="llama3.2", base_url="http://localhost:11434/v1", env=None),
    "anthropic": dict(kind="anthropic", model="claude-haiku-5-5", base_url=None, env="ANTHROPIC_API_KEY"),
}

SYSTEM_PROMPT = (
    "You are a careful document question-answering assistant.\n"
    "Rules:\n"
    "1. Answer ONLY from the numbered context passages given by the user.\n"
    "2. If the answer is not in the context, reply exactly: I could not find this in the documents.\n"
    "3. Keep the answer short (1-4 sentences) and cite the passage numbers like [1] or [2].\n"
    "4. Never invent facts, numbers, dates or names. Do not give legal or medical advice."
)


class LLM:
    def __init__(self, provider, model=None, api_key=None):
        cfg = PROVIDERS[provider]
        self.kind, self.provider = cfg["kind"], provider
        self.model = model or cfg["model"]
        key = api_key or (os.getenv(cfg["env"]) if cfg["env"] else None) or ("ollama" if "ollama" in provider else None)
        if self.kind == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=key, base_url=cfg["base_url"])
        elif self.kind == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=key)

    def complete(self, system, user, retries=4):
        for attempt in range(retries):
            try:
                if self.kind == "openai":
                    r = self.client.chat.completions.create(model=self.model, temperature=0.1, max_tokens=500,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                    return r.choices[0].message.content.strip()
                r = self.client.messages.create(model=self.model, max_tokens=500, temperature=0.1, system=system,
                                                messages=[{"role": "user", "content": user}])
                return "".join(b.text for b in r.content if getattr(b, "text", None)).strip()
            except Exception as e:                      # rate limit / overload -> wait and retry
                msg = str(e).lower()
                if attempt < retries - 1 and any(t in msg for t in ("429", "rate", "overload", "503", "timeout")):
                    time.sleep(5 * (attempt + 1)); continue
                raise
