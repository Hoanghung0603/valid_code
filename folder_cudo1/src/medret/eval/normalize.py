"""Text normalisation applied to both sides before tokenising for chunk matching.

docs/about-competition/evaluation.md (version of 2026-10-10), in this order: Unicode NFKC, HTML entity
decoding, lower case, collapse runs of whitespace, strip. Nothing else (no tag stripping, no punctuation
folding): the earlier spec listed "equivalent punctuation", the current one does not.
"""

from __future__ import annotations

import html
import re
import unicodedata

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = text.lower()
    return _WS_RE.sub(" ", text).strip()
