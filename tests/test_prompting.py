from litjev.api import McqQuestion, McqRequest
from litjev.prompting import build_decision_messages


def test_prompt_catalog_contains_all_ten_questions_and_options() -> None:
    questions = [
        McqQuestion(
            question_id=f"answer_{index}",
            prompt=f"Unique problem {index}",
            options={label: str(i) for i, label in enumerate("ABCDE")},
        )
        for index in range(1, 11)
    ]
    batch = McqRequest(questions=questions)

    messages = build_decision_messages("Choose", batch.to_schema())
    prompt = "\n".join(message["content"] for message in messages)

    assert "Unique problem 1" in prompt
    assert "Unique problem 10" in prompt
    assert "answer_1" in prompt
    assert "A." in prompt
    assert "E." in prompt
    assert prompt.startswith(
        "Answer independent multiple-choice questions. You will be asked for one field. "
    )
