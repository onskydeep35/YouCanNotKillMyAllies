from typing import Any, Dict, Type, Set
from pydantic import BaseModel
import json


class PydanticSchemaUtils:
    """
    Static utilities for converting Pydantic models into
    LLM-friendly descriptive JSON schemas.

    - Respects Field(exclude=True)
    - Respects model_config['exclude']
    - Uses field descriptions
    - Emits TYPE information (not example values)
    - Resolves nested Pydantic models via $ref
    """

    # ==========================================================
    # Public API
    # ==========================================================

    @staticmethod
    def to_descriptive_json(
        model: Type[BaseModel],
        *,
        include_descriptions: bool = True,
    ) -> Dict[str, Any]:
        schema = model.model_json_schema()
        defs = schema.get("$defs", {})
        excluded_fields = PydanticSchemaUtils._collect_excluded_fields(model)

        return PydanticSchemaUtils._build_example_from_schema(
            schema=schema,
            defs=defs,
            excluded_fields=excluded_fields,
            include_descriptions=include_descriptions,
        )

    @staticmethod
    def to_descriptive_pretty_json(
        model: Type[BaseModel],
        *,
        include_descriptions: bool = True,
        indent: int = 2,
        sort_keys: bool = False,
    ) -> str:
        example = PydanticSchemaUtils.to_descriptive_json(
            model,
            include_descriptions=include_descriptions,
        )

        return json.dumps(
            example,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=False,
        )

    # ==========================================================
    # Internal helpers
    # ==========================================================

    @staticmethod
    def _build_example_from_schema(
        *,
        schema: Dict[str, Any],
        defs: Dict[str, Any],
        excluded_fields: Set[str],
        include_descriptions: bool,
    ) -> Any:
        schema_type = schema.get("type")

        # ---------- Objects ----------
        if schema_type == "object":
            result: Dict[str, Any] = {}
            properties = schema.get("properties", {})

            for field_name, field_schema in properties.items():
                if field_name in excluded_fields:
                    continue

                if include_descriptions and "description" in field_schema:
                    result[field_name] = {
                        "_description": field_schema["description"],
                        "_type": PydanticSchemaUtils._schema_type_repr(
                            field_schema, defs
                        ),
                    }
                else:
                    result[field_name] = PydanticSchemaUtils._schema_type_repr(
                        field_schema, defs
                    )

            return result

        # ---------- Arrays ----------
        if schema_type == "array":
            return PydanticSchemaUtils._schema_type_repr(schema, defs)

        # ---------- Primitives ----------
        return PydanticSchemaUtils._schema_type_repr(schema, defs)

    @staticmethod
    def _schema_type_repr(schema: Dict[str, Any], defs: Dict[str, Any]) -> str:
        # ---------- Pydantic model reference ----------
        if "$ref" in schema:
            ref = schema["$ref"]
            # Example: "#/$defs/Problem"
            if ref.startswith("#/$defs/"):
                return ref.split("/")[-1]
            return "object"

        # ---------- Optional / Union ----------
        if "anyOf" in schema:
            return " | ".join(
                PydanticSchemaUtils._schema_type_repr(s, defs)
                for s in schema["anyOf"]
            )

        t = schema.get("type")

        if t == "string":
            return "string"
        if t == "integer":
            return "integer"
        if t == "number":
            return "number"
        if t == "boolean":
            return "boolean"
        if t == "array":
            item = schema.get("items", {})
            return f"array<{PydanticSchemaUtils._schema_type_repr(item, defs)}>"
        if t == "object":
            return "object"

        return "unknown"

    @staticmethod
    def _collect_excluded_fields(model: Type[BaseModel]) -> Set[str]:
        excluded: Set[str] = set()

        for name, field in model.model_fields.items():
            if field.exclude:
                excluded.add(name)

        model_exclude = model.model_config.get("exclude")
        if isinstance(model_exclude, (set, list, tuple)):
            excluded.update(model_exclude)

        return excluded
