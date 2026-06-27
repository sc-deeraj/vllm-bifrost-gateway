import httpx

from .config import settings


class VLLMError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class VLLMClient:
    """Thin wrapper around a single vLLM OpenAI-compatible server: the
    runtime LoRA admin endpoints (load/unload) plus a raw passthrough for
    inference requests. Owns no state of its own -- Registry is the source
    of truth for what *should* be loaded.
    """

    def __init__(self, base_url: str = settings.vllm_base_url):
        self._client = httpx.AsyncClient(
            base_url=base_url, timeout=settings.http_timeout_seconds
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> dict:
        resp = await self._client.get("/v1/models")
        resp.raise_for_status()
        return resp.json()

    async def load_lora_adapter(self, name: str, path: str) -> None:
        resp = await self._client.post(
            "/v1/load_lora_adapter",
            json={"lora_name": name, "lora_path": path},
        )
        if resp.status_code != 200:
            raise VLLMError(
                resp.status_code, f"vLLM rejected load of adapter '{name}': {resp.text}"
            )

    async def unload_lora_adapter(self, name: str) -> None:
        resp = await self._client.post(
            "/v1/unload_lora_adapter",
            json={"lora_name": name},
        )
        if resp.status_code != 200:
            raise VLLMError(
                resp.status_code,
                f"vLLM rejected unload of adapter '{name}': {resp.text}",
            )

    def stream(self, method: str, path: str, content: bytes, headers: dict):
        """Returns an httpx streaming context manager for passthrough proxying."""
        forward_headers = {
            k: v
            for k, v in headers.items()
            if k.lower() not in {"host", "content-length", "connection"}
        }
        return self._client.stream(
            method, path, content=content, headers=forward_headers
        )


vllm_client = VLLMClient()
