from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Provider Priority ---
    # Comma-separated: tries first provider, falls back on rate-limit/error
    # Only providers with keys configured will be used.
    # Example: "groq,gemini,openrouter"
    llm_provider_priority: str = "groq,gemini,openrouter"

    # --- Groq (free tier, no card; rate-limited) ---
    # Live free IDs: openai/gpt-oss-20b (fast/cheap), openai/gpt-oss-120b (flagship).
    # llama-3.3-70b-versatile was shut down Aug 2026.
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"

    # --- Gemini (free via AI Studio) ---
    # gemini-2.5-flash is gated to legacy users; 3.5-flash-lite is the free default.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"

    # --- OpenRouter (free; no card) ---
    # openrouter/free auto-routes to a rotating free model so single IDs
    # going paid/rotating don't break us. Comma-separated, tried in order.
    openrouter_api_key: str = ""
    openrouter_models: str = "openrouter/free"

    # Paths
    model_path: str = "models/model.pkl"

    # Logging
    log_level: str = "INFO"


settings = Settings()
