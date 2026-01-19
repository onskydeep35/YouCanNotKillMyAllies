from dataclasses import dataclass, field
from typing import Dict
from uuid import uuid4


@dataclass
class RunContext:
    run_id: str = field(default_factory=lambda: uuid4().hex)

    # Only durable role info
    final_roles: Dict[str, str] = field(default_factory=dict)
