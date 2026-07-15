"""Token-to-display-text conversion: byte-level BPE, SentencePiece, tiktoken bytes, passthrough.

The two stubs below deliberately reproduce two *different* real behaviors of
convert_tokens_to_string, confirmed against the actual bundled tokenizers
(web/public/tokenizer/{EleutherAI/pythia-70m,mapneo-250m}): GPT-2-style byte-level BPE already
returns a real leading space for a single marked token ("Ġcapital" -> " capital"), while
SentencePiece drops a lone token's leading "▁" instead of converting it ("▁world" -> "world", but
["is", "▁world"] -> "is world" — it only strips a marker that lands at the very start of the
reconstructed string). An earlier version of this test used one naive stub for both families and
asserted the wrong expected output for SentencePiece, silently passing while token_display_text's
predecessor shipped a real regression (a dropped leading space) for exactly this case.
"""

from pathlib import Path

import pytest

from lenslapse.sources import token_display_text

_MAPNEO_TOKENIZER = Path(__file__).resolve().parent.parent.parent / "web/public/tokenizer/mapneo-250m"


class _ByteLevelBPEStub:
    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens).replace("Ġ", " ")


class _SentencePieceStub:
    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        s = "".join(tokens).replace("▁", " ")
        return s[1:] if s.startswith(" ") else s


def test_byte_level_bpe_marker_becomes_a_real_space() -> None:
    assert token_display_text(_ByteLevelBPEStub(), "Ġcapital") == " capital"


def test_sentencepiece_marker_is_reattached_after_the_conversion_drops_it() -> None:
    assert token_display_text(_SentencePieceStub(), "▁world") == " world"
    assert token_display_text(_SentencePieceStub(), "▁中国") == " 中国"


def test_sentencepiece_non_initial_token_is_unaffected() -> None:
    assert token_display_text(_SentencePieceStub(), "world") == "world"


def test_plain_ascii_token_is_unaffected() -> None:
    assert token_display_text(_ByteLevelBPEStub(), "The") == "The"


def test_tiktoken_bytes_are_utf8_decoded_before_conversion() -> None:
    assert token_display_text(_ByteLevelBPEStub(), "中国的".encode()) == "中国的"


def test_invalid_utf8_bytes_decode_lossily_instead_of_raising() -> None:
    assert token_display_text(_ByteLevelBPEStub(), b"\xff\xfe") == "��"


def test_id_with_no_vocab_entry_falls_back_to_placeholder() -> None:
    """convert_ids_to_tokens returns None for ids beyond the tokenizer's real vocab size (e.g. a
    model's embedding matrix padded past it); matches the frontend's own '?' fallback."""
    assert token_display_text(_ByteLevelBPEStub(), None) == "?"


@pytest.mark.skipif(
    not _MAPNEO_TOKENIZER.is_dir(), reason="bundled mapneo-250m tokenizer not present in this checkout"
)
def test_real_sentencepiece_tokenizer_keeps_the_leading_space() -> None:
    """Regression test for the bug the stubs above were rewritten to catch: a hand-rolled stub is
    only as good as its fidelity to the real tokenizer, and the original stub here asserted the
    wrong answer for a real SentencePiece tokenizer. Exercises the actual bundled artifact."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(_MAPNEO_TOKENIZER), trust_remote_code=True)
    (word_initial_id,) = tok("中国", add_special_tokens=False)["input_ids"][:1]
    raw = tok.convert_ids_to_tokens([word_initial_id])[0]
    assert raw.startswith("▁")
    assert token_display_text(tok, raw).startswith(" ")
