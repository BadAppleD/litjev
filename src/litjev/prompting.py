import json

from litjev.schema import DecisionSchema


def build_decision_messages(state: str, schema: DecisionSchema) -> list[dict[str, str]]:
    catalog = {
        name: {"question": field.description, "choices": field.choices}
        for name, field in schema.items()
    }
    return [
        {
            "role": "system",
            "content": (
                "Answer independent multiple-choice questions. You will be asked for one field. "
                "Return only its exact choice label. Do not explain or reason aloud.\n"
                + json.dumps(catalog, ensure_ascii=False)
            ),
        },
        {"role": "user", "content": state},
    ]
