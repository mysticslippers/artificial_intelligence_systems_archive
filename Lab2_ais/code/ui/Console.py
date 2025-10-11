from typing import Dict, Any


class Console:
    def __init__(self):
        self.filters: Dict[str, Any] = {}

    def reset(self) -> None:
        self.filters.clear()