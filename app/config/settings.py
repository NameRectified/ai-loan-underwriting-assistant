from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Provider Priority ---
    # Comma-separated: tries first provider, falls back on rate-limit/error
    # Only providers with keys configured will be used.
    # Example: "groq,gemini,openrouter"
    llm_provider_priority: str = "groq,gemini,openrouter"

    # --- Groq ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # --- Gemini ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # --- OpenRouter ---
    openrouter_api_key: str = ""
    # Multiple models supported here because OpenRouter routes to different
    # underlying providers. If one model is rate-limited, the next may work.
    # Comma-separated, tried in order.
    openrouter_models: str = (
        "google/gemini-2.0-flash-001,"
        "meta-llama/llama-3-70b-instruct,"
        "mistralai/mistral-7b-instruct"
    )

    # Paths
    model_path: str = "models/model.pkl"
    storage_path: str = "data/applications.json"

    # Logging
    log_level: str = "INFO"


settings = Settings()
