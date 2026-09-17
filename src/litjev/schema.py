from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

MAX_CHOICES = 255
BOOLEAN_CHOICES = ("true", "false")
FieldType = Literal["enum", "boolean"]


@dataclass(frozen=True, slots=True)
class DecisionField:
    name: str
    field_type: FieldType
    description: str
    choices: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Field name must not be empty")
        if not self.description.strip():
            raise ValueError(f"Field '{self.name}' must have a description")
        if not self.choices:
            raise ValueError(f"Field '{self.name}' must define at least one choice")
        if len(self.choices) > MAX_CHOICES:
            raise ValueError(f"Field '{self.name}' exceeds the 255-choice limit")
        if len(set(self.choices)) != len(self.choices):
            raise ValueError(f"Field '{self.name}' choices must be unique")
        if any(not choice for choice in self.choices):
            raise ValueError(f"Field '{self.name}' choices must not be empty")

    @classmethod
    def enum(
        cls,
        name: str,
        description: str,
        choices: Sequence[str],
    ) -> DecisionField:
        return cls(name, "enum", description, tuple(str(choice) for choice in choices))

    @classmethod
    def boolean(cls, name: str, description: str) -> DecisionField:
        return cls(name, "boolean", description, BOOLEAN_CHOICES)


class DecisionSchema(Mapping[str, DecisionField]):
    def __init__(self, fields: Mapping[str, DecisionField]) -> None:
        if not fields:
            raise ValueError("Schema must contain at least one field")
        if any(name != field.name for name, field in fields.items()):
            raise ValueError("Schema keys must match field names")
        self._fields = MappingProxyType(dict(fields))

    @classmethod
    def from_mapping(cls, schema: Mapping[str, Mapping[str, Any]]) -> DecisionSchema:
        fields: dict[str, DecisionField] = {}
        for name, spec in schema.items():
            field_type = str(spec.get("type", "enum")).lower()
            description = str(spec.get("description", ""))
            if field_type == "boolean":
                field = DecisionField.boolean(name, description)
            elif field_type in {"enum", "choice", "selection"}:
                choices = spec.get("choices")
                if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
                    raise ValueError(f"Enum field '{name}' must provide a choices array")
                field = DecisionField.enum(name, description, choices)
            else:
                raise ValueError(f"Unsupported field type '{field_type}'")
            fields[name] = field
        return cls(fields)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._fields)

    def __getitem__(self, name: str) -> DecisionField:
        return self._fields[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)
