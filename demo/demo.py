#!/usr/bin/env python3
"""End-to-end proof of the acceptance criteria:

  (a) an OpenAI SDK call and an Anthropic SDK call both succeed against the
      same backend, through Bifrost
  (b) switching from Model A to Model B at runtime, no redeploy
  (c) a request for an unloaded model triggers load and then succeeds
  (d) admission control evicts the LRU adapter when capacity is full

Run after `docker compose up` and `scripts/make_demo_loras.py`.
"""

import os
import time

import httpx
from anthropic import Anthropic
from openai import OpenAI

BIFROST_URL = os.environ.get("BIFROST_URL", "http://localhost:8080")
MM_URL = os.environ.get("MM_URL", "http://localhost:9000")
PROVIDER = "local-vllm"
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "qwen2.5-3b-instruct")
ADAPTERS = ["demo-pirate", "demo-json", "demo-haiku"]

openai_client = OpenAI(base_url=f"{BIFROST_URL}/openai", api_key="dummy-key")
anthropic_client = Anthropic(base_url=f"{BIFROST_URL}/anthropic", api_key="dummy-key")
admin = httpx.Client(base_url=MM_URL, timeout=120.0)


def banner(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def model_name(adapter_id):
    return f"{PROVIDER}/{adapter_id}"


def show_registry():
    resp = admin.get("/admin/models")
    resp.raise_for_status()
    data = resp.json()
    print(f"capacity={data['capacity']} loaded_count={data['loaded_count']}")
    for m in data["models"]:
        print(f"  {m['id']:<14} state={m['state']:<9} pinned={m['pinned']} in_flight={m['in_flight']}")


def register_demo_adapters():
    banner("Setup: registering demo LoRA adapters with the Model Manager")
    for adapter_id in ADAPTERS:
        resp = admin.post("/admin/models", json={"id": adapter_id, "path": adapter_id})
        if resp.status_code == 200:
            print(f"registered '{adapter_id}'")
        elif resp.status_code == 409:
            print(f"'{adapter_id}' already registered")
        else:
            resp.raise_for_status()
    show_registry()


def step_a_dual_sdk():
    banner("(a) OpenAI SDK and Anthropic SDK both succeed against the same backend")
    question = "In one short sentence, what is vLLM?"

    oa_resp = openai_client.chat.completions.create(
        model=model_name(BASE_MODEL_ID),
        messages=[{"role": "user", "content": question}],
        max_tokens=60,
    )
    print(f"[OpenAI SDK]    {oa_resp.choices[0].message.content.strip()}")

    an_resp = anthropic_client.messages.create(
        model=model_name(BASE_MODEL_ID),
        max_tokens=60,
        messages=[{"role": "user", "content": question}],
    )
    print(f"[Anthropic SDK] {an_resp.content[0].text.strip()}")


def ask(adapter_id, question="What is the capital of Japan?"):
    resp = openai_client.chat.completions.create(
        model=model_name(adapter_id),
        messages=[{"role": "user", "content": question}],
        max_tokens=60,
    )
    return resp.choices[0].message.content.strip()


def step_b_runtime_switch():
    banner("(b) Switching models at runtime, no redeploy")
    print("-> calling 'demo-pirate'")
    print(ask("demo-pirate"))
    print("-> calling 'demo-json' immediately after, same running containers")
    print(ask("demo-json"))
    show_registry()


def step_c_cold_start():
    banner("(c) Cold start: requesting an adapter that isn't currently loaded")
    resp = admin.get("/admin/models").json()
    cold = next(m["id"] for m in resp["models"] if m["id"] == "demo-haiku")
    state_before = next(m["state"] for m in resp["models"] if m["id"] == "demo-haiku")
    print(f"'demo-haiku' state before request: {state_before}")

    started = time.monotonic()
    answer = ask("demo-haiku", "Tell me about the ocean.")
    elapsed = time.monotonic() - started
    print(f"response in {elapsed:.2f}s: {answer}")
    show_registry()


def step_d_admission_control():
    banner("(d) Admission control: capacity is full, LRU adapter gets evicted")
    print("capacity is 2; 'demo-haiku' just got loaded which should have evicted")
    print("whichever of pirate/json was least-recently-used:\n")
    show_registry()
    print(
        "\nnote the evicted adapter shows state=unloaded -- it was drained and "
        "unloaded from vLLM automatically, not just forgotten from the registry."
    )


if __name__ == "__main__":
    register_demo_adapters()
    step_a_dual_sdk()
    step_b_runtime_switch()
    step_c_cold_start()
    step_d_admission_control()
    banner("Done")
