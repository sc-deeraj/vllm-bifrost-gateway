#!/usr/bin/env python3
"""Train a few tiny, fast LoRA adapters against the platform's base model so
the demo has something real to hot-swap between -- proves the lifecycle
mechanism (load/unload/switch), not fine-tuning quality. Each adapter trains
in a couple of minutes on a single GPU.

Usage:
    python scripts/make_demo_loras.py --base-model Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

GENERIC_QUESTIONS = [
    "What is the capital of France?",
    "How do I boil an egg?",
    "What's a good way to learn programming?",
    "Tell me about the weather today.",
    "What is 12 times 8?",
    "Recommend a book to read.",
    "How does photosynthesis work?",
    "What should I eat for breakfast?",
    "Explain what gravity is.",
    "Give me advice on saving money.",
    "What's the tallest mountain in the world?",
    "How do airplanes fly?",
    "What is the meaning of life?",
    "Describe your ideal vacation.",
]

ADAPTERS = {
    "demo-pirate": {
        "description": "Always answers in exaggerated pirate speak.",
        "answer": lambda q: (
            f"Arrr, ye be askin' '{q}' Listen here, matey: "
            "I be answerin' true, but always in the tongue of the sea! Yarrr!"
        ),
    },
    "demo-json": {
        "description": "Always answers as a strict single-line JSON object.",
        "answer": lambda q: f'{{"question": "{q}", "answer": "A concise factual answer.", "style": "json"}}',
    },
    "demo-haiku": {
        "description": "Always answers as a three-line haiku.",
        "answer": lambda q: "Quiet mind reflects\non the question that you asked\nwisdom finds its form",
    },
}


def build_examples(tokenizer, answer_fn):
    examples = []
    for question in GENERIC_QUESTIONS:
        prompt_messages = [{"role": "user", "content": question}]
        full_messages = prompt_messages + [{"role": "assistant", "content": answer_fn(question)}]

        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True
        )
        full_ids = tokenizer.apply_chat_template(
            full_messages, tokenize=True, add_generation_prompt=False
        )
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        examples.append({"input_ids": full_ids, "labels": labels})
    return examples


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def pad_collate(batch, pad_token_id):
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids, labels, attention_mask = [], [], []
    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
        labels.append(ex["labels"] + [-100] * pad_len)
        attention_mask.append([1] * len(ex["input_ids"]) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }


def train_one_adapter(base_model, tokenizer, name, spec, out_dir, rank, epochs):
    print(f"\n=== training adapter '{name}': {spec['description']} ===")
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    examples = build_examples(tokenizer, spec["answer"])
    dataset = ListDataset(examples)

    adapter_out = os.path.join(out_dir, name)
    args = TrainingArguments(
        output_dir=os.path.join(adapter_out, "_checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=lambda batch: pad_collate(batch, tokenizer.pad_token_id),
    )
    trainer.train()

    model.save_pretrained(adapter_out)
    print(f"saved adapter to {adapter_out}")

    # get_peft_model() injects LoRA layers into base_model's modules in
    # place. unload() restores the original Linear layers so the next
    # adapter trains from a clean base instead of stacking on this one.
    return model.unload()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "loras"))
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--only", nargs="*", choices=list(ADAPTERS), help="train only these adapters")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"loading base model {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )

    targets = args.only or list(ADAPTERS)
    for name in targets:
        base_model = train_one_adapter(
            base_model, tokenizer, name, ADAPTERS[name], out_dir, args.rank, args.epochs
        )

    print("\nAll adapters trained. Register them with the Model Manager, e.g.:")
    for name in targets:
        print(
            f'  curl -X POST localhost:9000/admin/models -H "content-type: application/json" '
            f"-d '{{\"id\": \"{name}\", \"path\": \"{name}\", \"rank\": {args.rank}, "
            f"\"description\": \"{ADAPTERS[name]['description']}\"}}'"
        )


if __name__ == "__main__":
    main()
