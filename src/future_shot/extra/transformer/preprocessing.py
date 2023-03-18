from typing import Dict, Any, TYPE_CHECKING, List, Union

from future_shot.data import FutureShotPreprocessing


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class TokenizationFutureShotPreprocessing(FutureShotPreprocessing):
    def __init__(
        self,
        tokenizer: Union["PreTrainedTokenizerBase", str],
        text_field: str = "text",
        max_length: int = 32,
    ) -> None:
        if isinstance(tokenizer, str):
            import transformers
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        else:
            self._tokenizer = tokenizer
        self._text_field = text_field
        self._max_length = max_length

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        texts: List[str] = batch[self._text_field]

        features = self._tokenizer(
            texts,
            max_length=self._max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask="attention_mask" in self._tokenizer.model_input_names,
            return_token_type_ids="token_type_ids" in self._tokenizer.model_input_names,
        )

        return {**batch, **features}
