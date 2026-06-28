"""Query a Bifrost-fronted local model through both the OpenAI and Anthropic SDKs.

All tweakable values live in the CONFIG block. Everything below it is pure-ish
functions wired together in main().
"""

import base64
import json
import pathlib
import urllib.request
from dataclasses import dataclass

from openai import OpenAI
from anthropic import Anthropic


# --------------------------------------------------------------------------- #
# CONFIG                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    # Bifrost config holding governance.virtual_keys[0].value (the VIRTUAL key,
    # NOT the sk-vllm- provider key). Resolved relative to this file.
    config_path: pathlib.Path = pathlib.Path(__file__).parent / "bifrost" / "config.json"

    # Bifrost gateway. OpenAI- and Anthropic-format requests hit different paths.
    openai_base_url: str = "http://localhost:8080/openai"
    anthropic_base_url: str = "http://localhost:8080/anthropic"

    # Through Bifrost, models are addressed as "<provider>/<model-id>". The
    # provider prefix is required for routing, and bifrost/config.json defines
    # one provider per wire format -- not a single shared "local-vllm" -- since
    # /openai and /anthropic each need their own base_provider_type. model_id
    # must match BASE_MODEL_ID.
    openai_provider: str = "vllm-openai-compatible"
    anthropic_provider: str = "vllm-anthropic-compatible"
    model_id: str = "qwen3.5-9b"

    # Generous cap: Qwen3.5 is a reasoning model and spends tokens "thinking"
    # before answering. A small cap gets consumed by thinking, leaving no answer.
    max_tokens: int = 512

    question: str = "What is the capital of italy? Please explain your reasoning before answering."

    # Toggles for the optional sections.
    show_reasoning: bool = False
    run_vision: bool = True

    # Vision: Qwen3.5-9B is multimodal (requires VLLM_LANGUAGE_MODEL_ONLY=false,
    # the default). Many hosts 403 server-side fetchers without a browser UA, so
    # we fetch with a UA and inline as base64 -- the server fetches nothing.
    image_url: str = (
        "https://static.vecteezy.com/system/resources/thumbnails/"
        "050/393/628/small/cute-curious-gray-and-white-kitten-in-a-long-shot-photo.jpg"
    )
    vision_prompt: str = "Describe this image in one sentence."
    image_user_agent: str = "Mozilla/5.0"

    @property
    def openai_model(self) -> str:
        return f"{self.openai_provider}/{self.model_id}"

    @property
    def anthropic_model(self) -> str:
        return f"{self.anthropic_provider}/{self.model_id}"


# --------------------------------------------------------------------------- #
# SETUP                                                                        #
# --------------------------------------------------------------------------- #
def load_virtual_key(config_path: pathlib.Path) -> str:
    """Read governance.virtual_keys[0].value so it can't be mistyped or drift."""
    cfg = json.loads(config_path.read_text())
    return cfg["governance"]["virtual_keys"][0]["value"]


def make_openai_client(base_url: str, vk: str) -> OpenAI:
    # Bifrost enforces auth via the x-bf-vk header.
    return OpenAI(base_url=base_url, api_key=vk)


def make_anthropic_client(base_url: str, vk: str) -> Anthropic:
    return Anthropic(base_url=base_url, api_key=vk)


# --------------------------------------------------------------------------- #
# QUERIES                                                                      #
# --------------------------------------------------------------------------- #
def ask_openai(client: OpenAI, model: str, question: str, max_tokens: int):
    """Return (answer, reasoning) from the OpenAI-format endpoint."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        max_tokens=max_tokens,
    )
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", "")
    return (msg.content or "").strip(), (reasoning or "").strip()


def ask_anthropic(client: Anthropic, model: str, question: str, max_tokens: int) -> str:
    """Return the answer text from the Anthropic-format endpoint.

    Content can be [ThinkingBlock, ..., TextBlock]; pull the text block (empty
    if max_tokens was too small to get past thinking).
    """
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": question}],
    )
    # print(resp)
    text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
    return text.strip()


def fetch_image_data_uri(url: str, user_agent: str) -> str:
    """Fetch an image with a browser UA and inline it as a base64 data URI."""
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=30) as r:
        ctype = r.headers.get_content_type() or "image/jpeg"
        encoded = base64.b64encode(r.read()).decode()
    return f"data:{ctype};base64,{encoded}"


def ask_openai_vision(
    client: OpenAI, model: str, prompt: str, image_data_uri: str, max_tokens: int
) -> str:
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            }
        ],
    )
    return (resp.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# MAIN                                                                         #
# --------------------------------------------------------------------------- #
def main(cfg: Config = Config()) -> None:
    vk = load_virtual_key(cfg.config_path)
    openai_client = make_openai_client(cfg.openai_base_url, vk)
    anthropic_client = make_anthropic_client(cfg.anthropic_base_url, vk)
    oa_answer, oa_reasoning = ask_openai(
        openai_client, cfg.openai_model, cfg.question, cfg.max_tokens
    )
    print("[OpenAI SDK]   ", oa_answer)
    if cfg.show_reasoning:
        print("[OpenAI reasoning]", oa_reasoning or "(no reasoning text -- raise max_tokens)")

    an_answer = ask_anthropic(anthropic_client, cfg.openai_model, cfg.question, cfg.max_tokens)
    print("[Anthropic SDK]", an_answer or "(no answer text -- raise max_tokens)")

    if cfg.run_vision:
        data_uri = fetch_image_data_uri(cfg.image_url, cfg.image_user_agent)
        vision_answer = ask_openai_vision(
            openai_client, cfg.openai_model, cfg.vision_prompt, data_uri, cfg.max_tokens
        )
        print("[OpenAI vision]", vision_answer)


if __name__ == "__main__":
    main()