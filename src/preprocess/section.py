import importlib.util
import re
from typing import Optional, Tuple

import spacy

SOCIAL_HISTORY_PATTERNS = [
    r"social history",
    r"social hx",
    r"soc hx",
    r"social",
]


def _regex_section(text: str) -> Optional[str]:
    lowered = text.lower()
    for pattern in SOCIAL_HISTORY_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            start = match.start()
            subsection = text[start:]
            return subsection
    return None


def _medspacy_section(text: str) -> Optional[str]:
    if importlib.util.find_spec("medspacy") is None:
        return None
    import medspacy
    from medspacy.section_detection import SectionRule

    nlp = spacy.blank("en")
    nlp.add_pipe("medspacy_sectionizer")
    sectionizer = nlp.get_pipe("medspacy_sectionizer")
    rules = [
        SectionRule("social_history", "social history", ["social history", "social hx", "soc hx"]),
    ]
    sectionizer.add(rules)
    doc = nlp(text)
    for section in doc._.sections:
        if section.category == "social_history":
            return section.text
    return None


def extract_social_history(text: str) -> Tuple[str, bool]:
    section = _medspacy_section(text)
    if section:
        return section, True
    regex_section = _regex_section(text)
    if regex_section:
        return regex_section, True
    return text, False
