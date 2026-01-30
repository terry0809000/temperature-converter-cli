from typing import List

from transformers import PreTrainedTokenizer


def chunk_text(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    stride: int = 128,
) -> List[str]:
    tokens = tokenizer(
        text,
        return_overflowing_tokens=True,
        max_length=max_length,
        stride=stride,
        truncation=True,
    )
    input_ids = tokens["input_ids"]
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
