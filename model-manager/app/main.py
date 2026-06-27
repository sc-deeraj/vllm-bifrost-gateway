import json
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import settings
from .registry import (
    AlreadyExistsError,
    CapacityError,
    ColdStartTimeoutError,
    NotFoundError,
    registry,
)
from .schemas import AdapterView, ModelsResponse, RegisterAdapterRequest
from .vllm_client import VLLMError, vllm_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_manager")

app = FastAPI(title="Model Manager")

# Bifrost clients address models as "local-vllm/<id>" to disambiguate
# provider routing. Whether Bifrost strips that prefix before forwarding
# upstream isn't pinned down by its docs, so normalize defensively here:
# strip it if present, and rewrite the outgoing body to match, so vLLM
# always sees the bare id it actually knows about either way.
_PROVIDER_PREFIX = "local-vllm/"


def _strip_provider_prefix(model_id: str) -> str:
    if model_id.startswith(_PROVIDER_PREFIX):
        return model_id[len(_PROVIDER_PREFIX):]
    return model_id


def _error(status_code: int, message: str, error_type: str, headers: dict | None = None):
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type}},
        headers=headers,
    )


def _to_view(adapter) -> AdapterView:
    return AdapterView(
        id=adapter.id,
        base_model=settings.base_model_id,
        path=adapter.path,
        rank=adapter.rank,
        state=adapter.state.value,
        pinned=adapter.pinned,
        in_flight=adapter.in_flight,
        last_used=adapter.last_used or None,
        description=adapter.description,
    )


@app.on_event("startup")
async def on_startup():
    await registry.reconcile()


@app.on_event("shutdown")
async def on_shutdown():
    await vllm_client.aclose()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    vllm_ok = await vllm_client.health()
    return {"status": "ok" if vllm_ok else "degraded", "vllm_reachable": vllm_ok}


# ---------------------------------------------------------------------------
# Admin API -- model lifecycle control plane
# ---------------------------------------------------------------------------


@app.get("/admin/models", response_model=ModelsResponse)
async def admin_list_models():
    adapters = registry.list()
    loaded = sum(1 for a in adapters if a.state.value == "loaded" and not a.pinned)
    return ModelsResponse(
        capacity=settings.adapter_capacity,
        loaded_count=loaded,
        models=[_to_view(a) for a in adapters],
    )


@app.post("/admin/models")
async def admin_register_model(req: RegisterAdapterRequest):
    try:
        adapter = registry.register(req.id, req.path, rank=req.rank, description=req.description)
    except AlreadyExistsError:
        return _error(409, f"'{req.id}' is already registered", "already_exists")
    return _to_view(adapter)


@app.post("/admin/models/{model_id}/load")
@app.post("/admin/models/{model_id}/wake")
async def admin_load_model(model_id: str):
    return await _do_load(model_id)


@app.post("/admin/models/{model_id}/unload")
@app.post("/admin/models/{model_id}/sleep")
async def admin_unload_model(model_id: str):
    return await _do_unload(model_id)


@app.delete("/admin/models/{model_id}")
async def admin_forget_model(model_id: str):
    try:
        await registry.forget(model_id)
    except NotFoundError:
        return _error(404, f"'{model_id}' is not registered", "not_found")
    except ValueError as exc:
        return _error(400, str(exc), "invalid_request_error")
    except ColdStartTimeoutError:
        return _error(503, f"'{model_id}' could not be drained for removal", "drain_timeout")
    return {"deleted": model_id}


async def _do_load(model_id: str):
    try:
        adapter = await registry.load(model_id)
    except NotFoundError:
        return _error(404, f"'{model_id}' is not registered", "not_found")
    except CapacityError as exc:
        return _error(503, str(exc), "capacity_exhausted", headers={"Retry-After": "5"})
    except ColdStartTimeoutError:
        return _error(
            503,
            f"timed out loading '{model_id}' after {settings.cold_start_timeout_seconds}s",
            "cold_start_timeout",
            headers={"Retry-After": "5"},
        )
    except VLLMError as exc:
        return _error(502, exc.message, "upstream_error")
    return _to_view(adapter)


async def _do_unload(model_id: str):
    try:
        adapter = await registry.unload(model_id)
    except NotFoundError:
        return _error(404, f"'{model_id}' is not registered", "not_found")
    except ValueError as exc:
        return _error(400, str(exc), "invalid_request_error")
    except ColdStartTimeoutError:
        return _error(503, f"'{model_id}' still in-flight past drain timeout", "drain_timeout")
    return _to_view(adapter)


# ---------------------------------------------------------------------------
# Inference proxy (data plane) -- this is what Bifrost's custom provider
# points at. Mirrors vLLM's OpenAI-compatible surface; the only thing it
# adds is "make sure the requested model is loaded before forwarding".
# ---------------------------------------------------------------------------


@app.get("/v1/models")
async def list_models():
    try:
        return await vllm_client.list_models()
    except Exception as exc:
        return _error(502, f"could not reach vLLM: {exc}", "upstream_error")


@app.post("/v1/chat/completions")
@app.post("/v1/completions")
async def proxy_completions(request: Request):
    body = await request.body()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return _error(400, "request body is not valid JSON", "invalid_request_error")

    model_id_raw = payload.get("model")
    if not model_id_raw:
        return _error(400, "request body missing required field 'model'", "invalid_request_error")
    model_id = _strip_provider_prefix(model_id_raw)
    if model_id != model_id_raw:
        payload["model"] = model_id
        body = json.dumps(payload).encode()

    try:
        await registry.ensure_loaded(model_id, claim=True)
    except NotFoundError:
        return _error(
            404,
            f"model '{model_id}' is not known to this server. "
            f"Register it first via POST /admin/models.",
            "model_not_found",
        )
    except CapacityError as exc:
        return _error(503, str(exc), "capacity_exhausted", headers={"Retry-After": "5"})
    except ColdStartTimeoutError:
        return _error(
            503,
            f"model '{model_id}' did not finish loading within "
            f"{settings.cold_start_timeout_seconds}s",
            "cold_start_timeout",
            headers={"Retry-After": "5"},
        )
    except VLLMError as exc:
        return _error(502, exc.message, "upstream_error")

    started = time.monotonic()
    upstream_path = request.url.path

    # Open the upstream call and peek the status/headers before streaming the
    # body back, so errors from vLLM (e.g. context-length exceeded) surface
    # with the right status code instead of always returning 200.
    upstream_ctx = vllm_client.stream(request.method, upstream_path, body, dict(request.headers))
    try:
        upstream = await upstream_ctx.__aenter__()
    except Exception as exc:
        registry.release(model_id)
        return _error(502, f"could not reach vLLM: {exc}", "upstream_error")
    response_headers = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() in {"content-type"}
    }

    async def relay_opened():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream_ctx.__aexit__(None, None, None)
            registry.release(model_id)
            logger.info(
                "proxied %s model=%s status=%s in %.2fs",
                upstream_path,
                model_id,
                upstream.status_code,
                time.monotonic() - started,
            )

    return StreamingResponse(
        relay_opened(),
        status_code=upstream.status_code,
        headers=response_headers,
    )
