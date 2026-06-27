#!/usr/bin/env python3
"""Quick smoke test for the Model Manager control plane.

Exercises the three things you asked for, in order:

  1. load model  -> POST /admin/models  (register) + POST /admin/models/{id}/load
  2. see model   -> GET  /admin/models  (admin view) + GET /v1/models (vLLM view)
  3. chat        -> POST /v1/chat/completions

Stdlib only (urllib) -- no pip install needed. Run it against a running stack
(`docker compose up`).

  python3 scripts/test_model_manager.py                 # chat with the base model
  python3 scripts/test_model_manager.py --adapter demo-pirate --path demo-pirate

NOTE: the Model Manager listens on port 9000 (not 9090). Override with --url or
the MM_URL env var if yours differs. Loading a LoRA adapter requires its files
to exist under the mounted /loras dir (see scripts/make_demo_loras.py).
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def http_json(base_url, method, path, body=None, timeout=120.0):
    """Minimal JSON HTTP call. Returns (status_code, parsed_json_or_text)."""
    url = base_url.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        status = exc.code
    except urllib.error.URLError as exc:
        sys.exit(f"\n[FAIL] could not reach {url}: {exc.reason}\n"
                 f"       Is the stack up? Check `docker compose ps` and the port.")
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


def banner(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def show_registry(base_url):
    status, data = http_json(base_url, "GET", "/admin/models")
    if status != 200:
        print(f"  [admin/models -> {status}] {data}")
        return
    print(f"  capacity={data['capacity']} loaded_count={data['loaded_count']}")
    for m in data["models"]:
        print(f"    {m['id']:<18} state={m['state']:<9} "
              f"pinned={m['pinned']} in_flight={m['in_flight']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("MM_URL", "http://localhost:9000"),
                        help="Model Manager base URL (default: http://localhost:9000)")
    parser.add_argument("--model", default=os.environ.get("BASE_MODEL_ID", "qwen2.5-3b-instruct"),
                        help="base model id to chat with when no --adapter is given")
    parser.add_argument("--adapter", default=os.environ.get("ADAPTER_ID"),
                        help="LoRA adapter id to register + load + chat with")
    parser.add_argument("--path", default=os.environ.get("ADAPTER_PATH"),
                        help="adapter path under /loras (defaults to the adapter id)")
    parser.add_argument("--prompt", default="In one short sentence, what is vLLM?",
                        help="chat prompt to send")
    args = parser.parse_args()

    base_url = args.url
    print(f"Model Manager: {base_url}")

    # ---- 0. health -------------------------------------------------------
    banner("0. Health check")
    status, data = http_json(base_url, "GET", "/health")
    print(f"  GET /health -> {status} {data}")
    if status != 200:
        sys.exit("[FAIL] Model Manager is not healthy; aborting.")

    # ---- 1. load model ---------------------------------------------------
    banner("1. Load model")
    if args.adapter:
        adapter_id = args.adapter
        adapter_path = args.path or adapter_id
        chat_model = adapter_id

        status, data = http_json(base_url, "POST", "/admin/models",
                                 {"id": adapter_id, "path": adapter_path})
        if status == 200:
            print(f"  registered '{adapter_id}' (path={adapter_path})")
        elif status == 409:
            print(f"  '{adapter_id}' already registered")
        else:
            sys.exit(f"  [FAIL] register -> {status} {data}")

        print(f"  loading '{adapter_id}' into vLLM ...")
        started = time.monotonic()
        status, data = http_json(base_url, "POST", f"/admin/models/{adapter_id}/load")
        if status != 200:
            sys.exit(f"  [FAIL] load -> {status} {data}\n"
                     f"         (adapter files must exist under /loras; "
                     f"see scripts/make_demo_loras.py)")
        print(f"  loaded in {time.monotonic() - started:.2f}s -> state={data.get('state')}")
    else:
        chat_model = args.model
        print(f"  no --adapter given; base model '{chat_model}' is always resident.")
        print("  (pass --adapter <id> --path <path> to load a LoRA adapter too.)")

    # ---- 2. see model ----------------------------------------------------
    banner("2. See models")
    print("  Admin registry (GET /admin/models):")
    show_registry(base_url)
    print("\n  vLLM-visible models (GET /v1/models):")
    status, data = http_json(base_url, "GET", "/v1/models")
    if status == 200:
        for m in data.get("data", []):
            print(f"    {m.get('id')}")
    else:
        print(f"    [{status}] {data}")

    # ---- 3. chat ---------------------------------------------------------
    banner(f"3. Chat with '{chat_model}'")
    print(f"  prompt: {args.prompt}")
    started = time.monotonic()
    status, data = http_json(base_url, "POST", "/v1/chat/completions", {
        "model": chat_model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": 80,
        "stream": False,
    })
    if status != 200:
        sys.exit(f"  [FAIL] chat -> {status} {data}")
    answer = data["choices"][0]["message"]["content"].strip()
    print(f"  response in {time.monotonic() - started:.2f}s:")
    print(f"  >> {answer}")

    banner("Done -- all steps passed")


if __name__ == "__main__":
    main()
