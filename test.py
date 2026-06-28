import json
import pathlib

from openai import OpenAI
from anthropic import Anthropic

# Load the "app-prod" VIRTUAL key straight from bifrost/config.json so it can
# never be mistyped or drift. Bifrost enforces auth via the x-bf-vk header;
# this is governance.virtual_keys[0].value -- NOT the sk-vllm- provider key.
_cfg = json.loads((pathlib.Path(__file__).parent / "bifrost" / "config.json").read_text())
VK = _cfg["governance"]["virtual_keys"][0]["value"]

# Through Bifrost, models are addressed as "<provider>/<model-id>". The provider
# prefix is required for routing; MODEL_ID must match BASE_MODEL_ID.
PROVIDER = "local-vllm"
MODEL_ID = "qwen3.5-9b"
MODEL = f"{PROVIDER}/{MODEL_ID}"

openai_client = OpenAI(
    base_url="http://localhost:8080/openai",
    api_key=VK,
    default_headers={"x-bf-vk": VK},
)
anthropic_client = Anthropic(
    base_url="http://localhost:8080/anthropic",
    api_key=VK,
    default_headers={"x-bf-vk": VK},
)

question = "What is capital of france?"

oa_resp = openai_client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": question}],
    max_tokens=80,
)
print("[OpenAI SDK]   ", oa_resp.choices[0].message.content.strip())

an_resp = anthropic_client.messages.create(
    model=MODEL,
    max_tokens=80,
    messages=[{"role": "user", "content": question}],
)
print("[Anthropic SDK]", an_resp.content[0].text.strip())

# Vision: Qwen3.5-9B is multimodal, so an image_url part is served natively.
# (Requires VLLM_LANGUAGE_MODEL_ONLY=false, the default.)
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/640px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

vision_resp = openai_client.chat.completions.create(
    model=MODEL,
    max_tokens=80,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": IMAGE_URL}},
            ],
        }
    ],
)
print("[OpenAI vision]", vision_resp.choices[0].message.content.strip())
