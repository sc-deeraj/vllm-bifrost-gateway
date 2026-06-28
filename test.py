import base64
import json
import pathlib
import urllib.request

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

# max_tokens is generous: Qwen3.5 is a reasoning model, so it spends tokens
# "thinking" before the answer -- a small cap gets fully consumed by thinking
# and leaves no answer text.
oa_resp = openai_client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": question}],
    max_tokens=512,
)
oa_msg = oa_resp.choices[0].message
oa_reasoning = getattr(oa_msg, "reasoning", None) or getattr(oa_msg, "reasoning_content", "")
print("[OpenAI SDK]   ", (oa_msg.content or "").strip())
# print("[OpenAI reasoning]", oa_reasoning.strip() or "(no reasoning text -- raise max_tokens)")

an_resp = anthropic_client.messages.create(
    model=MODEL,
    max_tokens=512,
    messages=[{"role": "user", "content": question}],
)
# Anthropic-format content can be [ThinkingBlock, ..., TextBlock]; pull the
# text block (empty if max_tokens was too small to get past thinking).
an_text = next((b.text for b in an_resp.content if getattr(b, "type", None) == "text"), "")
print("[Anthropic SDK]", an_text.strip() or "(no answer text -- raise max_tokens)")

# Vision: Qwen3.5-9B is multimodal, so an image_url part is served natively.
# (Requires VLLM_LANGUAGE_MODEL_ONLY=false, the default.)
# IMAGE_URL = "https://static.vecteezy.com/system/resources/thumbnails/050/393/628/small/cute-curious-gray-and-white-kitten-in-a-long-shot-photo.jpg"

# # Many image hosts (Wikimedia, vecteezy, ...) return 403 to server-side fetchers
# # that lack a browser User-Agent, so vLLM can't pull the URL itself. Fetch it
# # here with a UA and inline it as a base64 data URI -- the server fetches nothing.
# _req = urllib.request.Request(IMAGE_URL, headers={"User-Agent": "Mozilla/5.0"})
# with urllib.request.urlopen(_req, timeout=30) as _r:
#     _ctype = _r.headers.get_content_type() or "image/jpeg"
#     IMAGE_DATA_URI = f"data:{_ctype};base64," + base64.b64encode(_r.read()).decode()

# vision_resp = openai_client.chat.completions.create(
#     model=MODEL,
#     max_tokens=512,
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "Describe this image in one sentence."},
#                 {"type": "image_url", "image_url": {"url": IMAGE_DATA_URI}},
#             ],
#         }
#     ],
# )
# print("[OpenAI vision]", (vision_resp.choices[0].message.content or "").strip())
