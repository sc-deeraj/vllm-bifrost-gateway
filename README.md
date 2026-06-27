# Self-hosted multi-model LLM platform (vLLM + Bifrost + Model Manager)

A single-GPU, Docker Compose deployment that serves one base model through
vLLM, fronts it with [Bifrost](https://github.com/maximhq/bifrost) for
OpenAI- and Anthropic-compatible API surfaces, virtual keys, and
observability, and uses a custom **Model Manager** to hot-swap LoRA
adapters at runtime with no redeploy.

## Decisions made (see spec section 5)

| Decision | Choice | Why |
|---|---|---|
| Deployment | Docker Compose, single node | One GPU box; no need for k8s scheduling overhead |
| GPU inventory | 1x RTX 4000, 24GB VRAM | Given; drives the base-model size and capacity defaults below |
| Model catalog / strategy | **Strategy A**: one base model + many LoRA adapters, hot-swapped | Matches "one base + many fine-tunes"; true zero-downtime, cheapest to operate on one GPU |
| Cold-start policy | Block-and-wait with a bounded timeout (default 90s), then 503 + `Retry-After` | LoRA loads are fast (sub-second to a few seconds); blocking briefly is simpler for clients than polling |
| Eviction policy | Strict LRU among non-pinned, non-busy adapters; drain in-flight requests first (default 30s) | Predictable, never kills a request mid-stream |
| Base model | `Qwen/Qwen2.5-3B-Instruct`, bf16, no quantization | Apache-2.0/ungated (no HF token friction), comfortably fits 24GB with headroom for KV cache + adapters, avoids LoRA+quantization compatibility edge cases |
| Auth | Bifrost virtual keys (`x-bf-vk` header) | Built into Bifrost OSS; no separate auth layer needed |

## Architecture

```
OpenAI SDK ──┐                                          ┌── vLLM (LoRA-enabled)
             ├─▶ Bifrost (/openai, /anthropic) ─▶ Model Manager ─▶  base model resident,
Anthropic SDK┘     virtual keys, budgets,           (proxy + admin)   adapters hot-swapped
                    observability, format
                    translation
```

**Bifrost** never sees model churn. Its `config.json` defines exactly one
custom provider, `local-vllm`, pointed at the Model Manager's stable
address. It only does what a gateway should: protocol translation, virtual
keys/budgets, rate limiting, observability. It does **not** know which LoRA
adapters exist or are loaded.

**Model Manager** (`model-manager/`) is both the control plane and the data
plane for the GPU:
- **Data plane**: exposes the same `/v1/chat/completions`, `/v1/completions`,
  `/v1/models` surface vLLM does, and transparently proxies to it -- after
  making sure the requested model is actually loaded. This is the layer that
  implements the cold-start policy.
- **Control plane**: `/admin/models*` for explicit register/load/unload, a
  registry of {adapter -> state, path, last-used}, and the
  capacity/eviction/drain logic.

**vLLM** runs one process, the base model permanently resident, with
`--enable-lora` and runtime LoRA updating turned on. It is **not** reachable
from outside the Docker network -- see [Security](#security).

### Why Bifrost's "register/deregister backends" requirement mostly disappears here

The general spec assumes Bifrost's routing table changes as backends come
and go (Strategy B/C: one vLLM process per model). Strategy A collapses that
churn entirely inside Model Manager -- there is only ever one backend
process, so Bifrost's config is static. If you outgrow one GPU and add a
second base model (Strategy C), that new vLLM instance would need its own
Model Manager-fronted address registered as a second Bifrost provider, using
Bifrost's provider-management REST API (`/providers`, see
`docs.getbifrost.ai`) to add it without restarting Bifrost. That extension
point is not built here because it isn't needed for one base model + LoRA
adapters -- see [Adding a new model](#adding-a-new-model).

## Hardware / versions

- 1x NVIDIA GPU, 24GB VRAM (RTX 4000 class)
- NVIDIA driver >= 550, CUDA >= 12.4, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host
- Docker Compose v2 (the `deploy.resources.reservations.devices` GPU syntax requires it)
- `vllm/vllm-openai:v0.9.2` (pinned); `maximhq/bifrost:latest` (pin to a released tag for production -- check Docker Hub)

## Setup

A `stack.sh` helper wraps Docker Compose (`./stack.sh help` lists everything):

```bash
cp .env.example .env         # optional: change base model / ports
./stack.sh up                # build + start vllm, model-manager, bifrost
./stack.sh logs vllm         # first start downloads the base model; wait for "Uvicorn running"
```

`./stack.sh restart` does a full stop -> rebuild -> up cycle; `./stack.sh down`
tears it down (named volumes kept). Plain `docker compose up -d --build` still
works if you prefer.

### Choosing the base model

`docker-compose.yml` defaults to `Qwen/Qwen2.5-3B-Instruct` (bf16). Override in
`.env` -- `BASE_MODEL` **must be a fully-qualified Hugging Face repo id**
(`org/name`; a bare name 401s):

```bash
# .env -- bigger model via AWQ 4-bit, fits a ~20GB card
BASE_MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ
BASE_MODEL_ID=qwen2.5-14b-instruct-awq
```

bf16 bases keep LoRA hot-swap simple; AWQ/GPTQ bases buy capacity but LoRA +
quantization can be finicky. Changing the base invalidates adapters trained
against the previous one. After editing `.env`, `./stack.sh up` (or
`docker compose up -d vllm`) to apply.

### Virtual keys (required -- auth is enforced)

Bifrost runs with `client.enforce_auth_on_inference: true`, so every inference
request **must** present a valid virtual key in the `x-bf-vk` header. Three are
pre-seeded in `bifrost/config.json` -- `app-prod`, `app-staging`, `data-team`,
each scoped to the `local-vllm` provider with all models allowed. Rotate them
there, or add more via the Bifrost UI (`http://localhost:8080` -> Governance ->
Virtual Keys) or its governance REST API. Example call:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-bf-vk: <one-of-the-virtual-keys>" \
  -d '{"model":"local-vllm/qwen2.5-3b-instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":20}'
```

> **Production:** move the virtual-key values and the upstream provider key out
> of the committed `bifrost/config.json` into env vars
> (`"value": "env.VK_APP_PROD"`) so secrets aren't checked into git.

### Demo

1. **Generate the demo LoRA adapters** (trains 3 tiny adapters against the
   same base model, a couple of minutes on the GPU):
   ```bash
   pip install -r scripts/requirements-train.txt   # torch, transformers, peft
   python scripts/make_demo_loras.py
   ```
   Or point `--out-dir`/register your own existing adapter under `./loras/<id>/`.
2. **Run the demo**:
   ```bash
   pip install -r demo/requirements.txt
   BASE_MODEL_ID=qwen2.5-3b-instruct python demo/demo.py
   ```
   This proves, against the running containers with no redeploy:
   - (a) an OpenAI SDK call and an Anthropic SDK call both succeed through Bifrost
   - (b) switching between two loaded LoRA adapters at runtime
   - (c) requesting a third, not-yet-loaded adapter triggers a cold-start load and succeeds
   - (d) loading that third adapter evicts the least-recently-used one (capacity is 2 by default)

   > `demo/demo.py` predates key enforcement and sends a placeholder key. Give
   > its OpenAI/Anthropic clients a real `x-bf-vk` (see `test.py` for the
   > pattern), or temporarily set `enforce_auth_on_inference: false` while
   > running it.

Clients address models as `<bifrost-provider>/<model-id>`, e.g.
`local-vllm/demo-pirate` or `local-vllm/qwen2.5-3b-instruct` for the base
model with no adapter.

## Model Manager admin API

| Method & path | Effect |
|---|---|
| `GET /admin/models` | List all known adapters + the base model, with state/capacity |
| `POST /admin/models` `{id, path, rank?, description?}` | Register a new adapter (path relative to `/loras`, or absolute) |
| `POST /admin/models/{id}/load` (alias `/wake`) | Load now; evicts the LRU adapter first if at capacity |
| `POST /admin/models/{id}/unload` (alias `/sleep`) | Unload now; drains in-flight requests first |
| `DELETE /admin/models/{id}` | Unload (if needed) and forget the adapter entirely |
| `GET /health` | Model Manager + vLLM reachability |

These run on the internal network only -- port 9000 is **not** published to the
host (see [Security](#security)). Call them from another compose service, or
from the host via `docker compose exec`, e.g.:

```bash
docker compose exec model-manager \
  python3 -c "import urllib.request,json; print(json.dumps(json.load(urllib.request.urlopen('http://localhost:9000/admin/models')),indent=2))"
```

`sleep`/`wake` are aliases of `unload`/`load` here, not separate states.
Strategy B's "sleep" (CPU-offload a multi-GB engine, keep the process alive)
doesn't apply to LoRA adapters -- they're MBs, not GBs, so there's no
meaningful intermediate state between "loaded in vLLM" and "known on disk
but not loaded." If you add Strategy B models later, that distinction
becomes real and vLLM's own `/sleep` + `/wake_up` (dev-mode) endpoints are
the right tool -- not built here since Strategy A doesn't need them.

## Edge case behavior (see spec section 4)

- **Cold start**: block-and-wait up to `MM_COLD_START_TIMEOUT_SECONDS`
  (default 90s), then `503` + `Retry-After: 5`.
- **Capacity full**: evict strict LRU among loaded, non-pinned, non-busy
  adapters. If the LRU candidate(s) are still serving requests, wait up to
  `MM_DRAIN_TIMEOUT_SECONDS` (default 30s) before trying the next
  candidate; if none can be drained in time, `503 capacity_exhausted`.
- **In-flight during eviction**: never killed -- eviction only calls
  `/v1/unload_lora_adapter` after the victim's in-flight counter hits zero
  or the drain deadline passes (in which case that candidate is skipped).
- **Load failure / OOM**: the adapter's state is left `unloaded` (the prior
  state), never stuck in a half-loaded state; the registry's catalog entry
  is untouched so it can be retried.
- **vLLM crash**: `GET /health` reports `degraded` (`vllm_reachable: false`)
  to whatever is health-checking Model Manager; Bifrost will surface
  connection errors as upstream errors until vLLM (Compose `restart:
  unless-stopped`) comes back, at which point the *next* Model Manager
  restart (or a future explicit reconcile) re-derives real state from
  vLLM's live `/v1/models`.
- **Model Manager restart**: `reconcile()` runs on startup -- loads the
  on-disk adapter catalog, then asks vLLM what's *actually* loaded right
  now and trusts that over any assumption.
- **Unknown model name**: fails fast with `404 model_not_found`, never
  blocks -- block-and-wait is only for known-but-unloaded models.
- **Concurrent admin ops**: one global lock serializes every state
  transition (load, unload, evict) and the in-flight claim that goes with
  it, so two concurrent requests for the same not-yet-loaded adapter
  coalesce onto one load instead of racing, and eviction can never unload an
  adapter out from under a request that just got told it was ready. Simpler
  than per-adapter locking and provably free of the deadlock/race a
  finer-grained version would risk; the tradeoff is that an unrelated
  already-loaded model's request can briefly queue behind a slow load/evict
  of some other model. Fine on one GPU at modest concurrency.

## Security

- vLLM's container port (8000) is **not published to the host** -- it's
  reachable only from the Compose-internal network, and only Model Manager
  talks to it. This matters because `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`
  lets any caller with access to that port load an arbitrary local path as
  a LoRA adapter; it must never be internet- or even LAN-facing.
- Model Manager's port (9000) is **not published to the host either** -- it's
  `expose`-only on the internal network (`http://model-manager:9000`), reached
  only by Bifrost. It's a control-plane surface with no auth of its own, so it
  stays off the host; for host-side admin use `docker compose exec model-manager ...`.
- **Bifrost (8080) is the only public surface, and auth is enforced**
  (`enforce_auth_on_inference: true`): inference requires a valid virtual key
  (`x-bf-vk`); unkeyed requests are rejected. Bifrost reaches the Model Manager
  over the Docker private network, which requires `allow_private_network: true`
  on the provider's `network_config` (Bifrost blocks RFC-1918 IPs by default).
- Bifrost stores the upstream provider key (which vLLM/Model Manager don't
  actually validate); it's never exposed to clients. Keep the real virtual-key
  and provider-key values in environment variables, not committed to
  `bifrost/config.json`.

## Adding a new model

**Another LoRA adapter on the same base model** (the common case): train it
against the exact same base model identifier as `BASE_MODEL`/`MM_BASE_MODEL`
(see `scripts/make_demo_loras.py` for a template), drop it under
`./loras/<id>/`, then `POST /admin/models {"id": "...", "path": "<id>"}`.
Nothing else changes -- no restart of any container.

**A second, different base model** (Strategy B or C): out of scope for this
build (chosen catalog is one base + LoRA adapters), but the extension path
is: stand up a second `vllm` service in `docker-compose.yml`, point a second
Model Manager instance (or extend this one to manage multiple base models)
at it, and register a second Bifrost custom provider via its runtime
provider-management REST API so it's reachable with no Bifrost restart --
exactly the "registers/deregisters backends in Bifrost" behavior the
general spec describes for catalogs that don't fit one base model.

## Known limitations

- Model Manager runs as a single process/worker -- its registry and locks
  are in-memory, not safe to scale to multiple replicas without
  externalizing state (e.g. Redis). For one GPU this is the right amount of
  complexity; don't add HA machinery you don't need yet.
- No Kubernetes manifests -- Compose was the explicit choice for this
  single-node deployment.
- `MM_ADAPTER_CAPACITY` (default 2) is set artificially low so eviction is
  easy to observe in the demo. Size it from real measurements
  (`--max-cpu-loras` on the vLLM side, and how much VRAM/RAM each adapter
  actually costs) before using this for anything real.
