import pytest

from litjev.schema import DecisionField, DecisionSchema


def test_schema_accepts_enum_and_boolean_fields() -> None:
    schema = DecisionSchema.from_mapping(
        {
            "answer_1": {
                "type": "enum",
                "description": "Choose an answer.",
                "choices": ["A", "B", "C"],
            },
            "needs_review": {
                "type": "boolean",
                "description": "Whether review is needed.",
            },
        }
    )

    assert schema["answer_1"].choices == ("A", "B", "C")
    assert schema["needs_review"].choices == ("true", "false")


def test_schema_rejects_duplicate_and_excessive_choices() -> None:
    with pytest.raises(ValueError, match="unique"):
        DecisionField.enum("answer", "Choose.", ["A", "A"])

    with pytest.raises(ValueError, match="255"):
        DecisionField.enum("answer", "Choose.", [str(index) for index in range(256)])
