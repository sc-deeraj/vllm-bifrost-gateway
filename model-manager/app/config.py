from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    vllm_base_url: str = "http://vllm:8000"
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    base_model_id: str = "qwen2.5-3b-instruct"

    lora_dir: str = "/loras"
    registry_file: str = "/loras/registry.json"

    # Max number of LoRA adapters Model Manager will keep loaded in vLLM at
    # once. Must be <= the vLLM server's --max-cpu-loras. Kept low by default
    # so admission control / eviction is easy to observe in the demo; in
    # production size this from real VRAM headroom per adapter.
    adapter_capacity: int = 2

    cold_start_timeout_seconds: float = 90.0
    drain_timeout_seconds: float = 30.0
    drain_poll_interval_seconds: float = 0.5

    http_timeout_seconds: float = 120.0

    class Config:
        env_prefix = "MM_"


settings = Settings()
