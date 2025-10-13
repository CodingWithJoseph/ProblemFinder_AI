"""Text normalisation helpers used across deduplication and classification."""

from __future__ import annotations

import html
import logging
import re
import unicodedata
from typing import Iterable, List, Sequence, Tuple
from urllib.parse import SplitResult, urlsplit

logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"https?://\S+")
MENTION_PATTERN = re.compile(r"/?u/[A-Za-z0-9_-]+")


def normalise_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim the string."""

    return re.sub(r"\s+", " ", text).strip()


def strip_markdown(text: str) -> str:
    """Remove common Markdown constructs from text."""

    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", lambda m: m.group(0).split("]")[0][1:], text)
    text = re.sub(r"^>+\s?", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^#{1,6}\s*", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^(?:\s*[-*+]\s+|\d+\.\s+)", " ", text, flags=re.MULTILINE)
    return text


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""

    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text)


def is_valid_ipv6_url(url: str) -> bool:
    """Return ``True`` if ``url`` contains a valid IPv6 literal."""

    ipv6_pattern = r'://\[([0-9a-fA-F:]+)\]'
    return re.search(ipv6_pattern, url) is not None


def safe_urlsplit(url: str) -> SplitResult | None:
    """Safely split ``url`` while tolerating IPv6 edge cases."""

    try:
        if not url or not isinstance(url, str):
            return None
        if "[" in url and "]" in url:
            if not is_valid_ipv6_url(url):
                logger.warning("Skipping invalid IPv6 URL: %s", url[:50])
                return None
        return urlsplit(url)
    except ValueError as exc:
        logger.warning("ValueError parsing URL %s: %s", url[:50], exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unexpected error parsing URL %s: %s", url[:50], exc)
        return None


def remove_urls(text: str) -> Tuple[str, List[str]]:
    """Remove URLs from ``text`` while collecting them."""

    urls: List[str] = []

    def _collect(match: re.Match[str]) -> str:
        url = match.group(0)
        parsed = safe_urlsplit(url)
        if parsed is not None:
            urls.append(url)
        else:
            logger.debug("Skipped malformed URL: %s", url[:50])
        return " "

    cleaned = URL_PATTERN.sub(_collect, text)
    return cleaned, urls


def remove_crosspost_boilerplate(text: str) -> str:
    """Strip common Reddit cross-post boilerplate lines."""

    return re.sub(r"cross\s*post(ed)?\s+from\s+r/\w+", " ", text, flags=re.IGNORECASE)


def remove_mentions(text: str) -> str:
    """Remove Reddit user mentions to stabilise dedupe tokens."""

    return MENTION_PATTERN.sub(" ", text)


def strip_emoji(text: str) -> str:
    """Remove emoji characters which add noise to similarity checks."""

    return "".join(ch for ch in text if unicodedata.category(ch) != "So")


def tokenise(text: str) -> List[str]:
    """Split ``text`` into lowercase tokens."""

    return [tok for tok in re.split(r"[^a-z0-9]+", text) if tok]


def char_ngrams(text: str, n: int = 5) -> List[str]:
    """Return character-level n-grams used in fuzzy similarity."""

    if len(text) < n:
        return [text]
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """Compute the Jaccard similarity between two sequences."""

    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def combined_similarity(text_a: str, text_b: str) -> float:
    """Blend token and character similarities for robust duplicate detection."""

    tokens_a = tokenise(text_a)
    tokens_b = tokenise(text_b)
    token_score = jaccard(tokens_a, tokens_b)

    chars_a = char_ngrams(text_a)
    chars_b = char_ngrams(text_b)
    char_score = jaccard(chars_a, chars_b)
    return 0.7 * token_score + 0.3 * char_score


def normalise_text_block(text: str) -> str:
    """Apply the canonical normalisation pipeline used for deduping."""

    lowered = text.lower()
    lowered = remove_crosspost_boilerplate(lowered)
    lowered = remove_mentions(lowered)
    no_html = strip_html(lowered)
    cleaned_urls_text, _ = remove_urls(no_html)
    no_markdown = strip_markdown(cleaned_urls_text)
    no_emoji = strip_emoji(no_markdown)
    no_symbols = re.sub(r"[^a-z0-9\s]", " ", no_emoji)
    return normalise_whitespace(no_symbols)
