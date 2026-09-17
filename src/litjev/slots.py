"""Compile independent full-prefix branches, preserving the original cached prompts."""

import json
from dataclasses import dataclass

from litjev.prompting import build_decision_messages

SLOT_FORMAT = "field_answer_v1"


@dataclass(frozen=True)
class CompiledSlots:
    input_ids: list[list[int]]
    positions: list[int]
    candidates: list[list[int]]
    slot_texts: list[str]
    slot_ids: list[list[int]]
    prefix_text: str
    prefix_length: int


def compile_slots(tokenizer, state, schema, max_input_tokens=16384):
    prefix = tokenizer.apply_chat_template(
        build_decision_messages(state, schema),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    prefix_length = len(prefix_ids)
    rows = []
    positions, candidates, texts, slot_ids = [], [], [], []
    for name, field in schema.items():
        text = f"Field {json.dumps(name)}\nAnswer:"
        tokens = tokenizer.encode(text, add_special_tokens=False)
        choices = []
        for choice in field.choices:
            appended = tokenizer.encode(text + " " + choice, add_special_tokens=False)
            if appended[:-1] != tokens or len(appended) != len(tokens) + 1:
                raise ValueError(f"Choice {choice!r} must be one token after its key slot")
            choices.append(appended[-1])
        if len(set(choices)) != len(choices):
            raise ValueError("Candidate token collision")
        # Exactly the original prefix IDs followed by this branch's suffix IDs.
        row = prefix_ids + tokens
        rows.append(row)
        positions.append(len(row) - 1)
        candidates.append(choices)
        texts.append(text)
        slot_ids.append(tokens)
    if max(map(len, rows)) > max_input_tokens:
        raise ValueError("Request exceeds input token limit; no truncation performed")
    return CompiledSlots(rows, positions, candidates, texts, slot_ids, prefix, prefix_length)
