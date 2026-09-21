import json


def state_text(state):
    return (
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, allow_nan=False)
    )


def build_decision_messages(state, schema=None):
    """Only shared state is prefetched; question ids never affect inference."""
    return [
        {
            "role": "system",
            "content": (
                "Evaluate the state using the question and labeled options that follow. "
                "Return only the option code. Do not explain or reason aloud."
            ),
        },
        {"role": "user", "content": state_text(state)},
    ]


def build_thinking_messages(state, body):
    """Slow path: the question joins the state in the user turn and reasoning is allowed."""
    return [
        {
            "role": "system",
            "content": (
                "Evaluate the state using the question and labeled options. "
                "Think it through first, then answer with only the option code."
            ),
        },
        {"role": "user", "content": state_text(state) + "\n\n" + body},
    ]


ANSWER_BOUNDARY = "\nAnswer:"


def question_body(question, codes):
    """The question text shared by the fast readout and the slow thinking path."""
    options = [
        {"code": codes[i], "option": key, "description": description}
        for i, (key, description) in enumerate(
            zip(question.choices, question.descriptions, strict=True)
        )
    ]
    body = {"type": question.type, "instructions": question.instructions, "options": options}
    return "Question: " + json.dumps(body, ensure_ascii=False, allow_nan=False)


def question_suffix(question, codes):
    return question_body(question, codes) + ANSWER_BOUNDARY
