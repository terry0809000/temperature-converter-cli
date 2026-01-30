import re
from typing import Literal

PHI_PATTERN = re.compile(r"\[\*\*.*?\*\*\]")


def handle_phi(text: str, strategy: Literal["replace", "remove"] = "replace") -> str:
    if strategy == "remove":
        return PHI_PATTERN.sub("", text)
    return PHI_PATTERN.sub("[PHI]", text)
