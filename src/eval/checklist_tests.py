from typing import Dict, List


def generate_negation(text: str) -> str:
    return text.replace("has", "does not have").replace("uses", "does not use")


def generate_historical(text: str) -> str:
    return "History of " + text


def generate_attribution(text: str) -> str:
    return "Patient denies but family reports " + text


def generate_misspelling(text: str) -> str:
    return text.replace("tobacco", "tobaco").replace("alcohol", "alcoh0l")


def build_checklist(texts: List[str]) -> Dict[str, List[str]]:
    return {
        "negation": [generate_negation(t) for t in texts],
        "historical": [generate_historical(t) for t in texts],
        "attribution": [generate_attribution(t) for t in texts],
        "misspelling": [generate_misspelling(t) for t in texts],
    }
