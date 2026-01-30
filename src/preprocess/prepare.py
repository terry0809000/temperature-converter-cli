from typing import Dict, List, Tuple

from transformers import AutoTokenizer

from src.preprocess.chunking import chunk_text
from src.preprocess.phi import handle_phi
from src.preprocess.section import extract_social_history


def preprocess_text(text: str, preprocess_cfg: Dict) -> Tuple[str, bool]:
    text = handle_phi(text, preprocess_cfg.get("phi_strategy", "replace"))
    if preprocess_cfg.get("use_social_history", True):
        section_text, found = extract_social_history(text)
        return section_text, found
    return text, False


def prepare_texts(
    texts: List[str],
    preprocess_cfg: Dict,
    model_name: str,
) -> Tuple[List[str], List[int]]:
    processed = []
    index_map = []
    for idx, text in enumerate(texts):
        section_text, _ = preprocess_text(text, preprocess_cfg)
        processed.append(section_text)
        index_map.append(idx)
    if preprocess_cfg.get("full_note", False):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        chunked = []
        chunk_map = []
        for idx, text in enumerate(processed):
            chunks = chunk_text(
                text,
                tokenizer,
                max_length=preprocess_cfg.get("max_length", 512),
                stride=preprocess_cfg.get("stride", 128),
            )
            chunked.extend(chunks)
            chunk_map.extend([idx] * len(chunks))
        return chunked, chunk_map
    return processed, index_map
