vllm bench serve --model qwen2.5-14b-instruct-awq \
  --base-url http://localhost:8000 --num-prompts 100 --request-rate 5