"""Lightweight, auditable ICD-10 chapter routing for medical retrieval.

The router deliberately produces multiple chapter hypotheses and an UNKNOWN
flag.  It is intended as a retrieval feature (candidate boost/union), not as a
hard clinical diagnosis or an unconditional corpus filter.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
ICD_CODE_RE = re.compile(r"\b([A-Z])\d{2}(?:\.\d+)?\b", re.IGNORECASE)
GENERIC_TERMS = {
    "a", "about", "after", "all", "also", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off",
    "on", "once", "only", "or", "our", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "you", "your", "yours",
    # Patient-question boilerplate and low-information ICD wording.
    "advice", "ask", "classified", "condition", "conditions", "consult", "consultation", "doctor",
    "elsewhere", "examination", "know", "mean", "please", "specified", "unspecified", "want",
}


def normalize_text(text: str) -> str:
    """Lowercase and remove accents so Vietnamese terms remain searchable."""

    normalized = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall(normalize_text(text))
        if len(token) > 1 and token not in GENERIC_TERMS
    ]


@dataclass(frozen=True)
class ICDEntry:
    code: str
    chapter: str
    text: str


@dataclass(frozen=True)
class ChapterRoute:
    chapter: str
    score: float


@dataclass(frozen=True)
class RoutingResult:
    routes: tuple[ChapterRoute, ...]
    confidence: str
    unknown: bool
    margin: float
    matched_terms: int


# Vietnamese surface forms are intentionally broad.  Specific disease names
# are supplied by the full English ICD descriptions through query_en.
DEFAULT_VI_LEXICON: Mapping[str, tuple[str, ...]] = {
    "01": ("nhiem trung", "truyen nhiem", "ky sinh trung", "virus", "vi khuan", "nam", "sot xuat huyet", "viem gan"),
    "02": ("ung thu", "u lanh", "u ac", "u tan sinh", "di can", "hoa tri", "xa tri"),
    "03": ("thieu mau", "bach cau", "tieu cau", "dong mau", "tan mau", "suy tuy", "mien dich"),
    "04": ("dai thao duong", "tieu duong", "tuyen giap", "noi tiet", "chuyen hoa", "beo phi", "suy dinh duong"),
    "05": ("tram cam", "lo au", "tam than", "hanh vi", "mat ngu", "tu ky", "stress", "hoang tuong"),
    "06": ("than kinh", "dong kinh", "dau nua dau", "parkinson", "alzheimer", "dot quy nao", "te bi"),
    "07": ("thi luc", "giac mac", "võng mac", "vong mac", "glocom", "duc thuy tinh the"),
    "08": ("thinh luc", "u tai", "viem tai", "xương chum", "xuong chum"),
    "09": ("tim", "mach mau", "huyet ap", "dong mach", "tinh mach", "suy tim", "nhoi mau"),
    "10": ("ho hap", "phoi", "phe quan", "hen phe quan", "kho tho", "viem mui", "viem xoang"),
    "11": ("tieu hoa", "da day", "thuc quan", "ruot", "dai trang", "viem gan", "tui mat", "tuyen tuy", "hau mon"),
    "12": ("viem da", "ngua da", "may day", "vay nen", "mun trung ca", "rụng toc", "rụng tóc", "benh mong"),
    "13": ("co xuong khop", "xuong", "khop", "cot song", "thoai hoa", "gout", "loang xuong", "dau lung"),
    "14": ("tiet nieu", "sinh duc", "benh than", "suy than", "bang quang", "tien liet tuyen", "am dao", "buong trung", "vo sinh"),
    "15": ("mang thai", "thai ky", "sinh de", "hau san", "san khoa", "beta hcg", "hcg", "kinh nguyet"),
    "16": ("so sinh", "chu sinh", "tre de non", "ngat luc sinh", "vang da so sinh"),
    "17": ("bam sinh", "di tat", "nhiem sac the", "down", "di truyen"),
    "18": ("trieu chung", "bat thuong xet nghiem", "chua ro nguyen nhan", "chong mat khong ro", "dau khong ro", "sot khong ro"),
    "19": ("chan thuong", "ngo doc", "bong nhiet", "gay xuong", "vet thuong", "con vat can", "di vat"),
    "20": ("tai nan", "te nga", "va dam", "duoi nuoc", "nguyen nhan ben ngoai"),
    "21": ("kham suc khoe", "tiem chung", "tu van", "tam soat", "theo doi", "phuc hoi", "ke hoach hoa"),
    "22": ("covid", "sars cov", "ma dac biet"),
}


class ICD10ChapterRouter:
    """BM25-over-ontology router with Vietnamese lexical evidence."""

    def __init__(
        self,
        entries: Sequence[ICDEntry],
        vi_lexicon: Mapping[str, Sequence[str]] | None = None,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        if not entries:
            raise ValueError("ICD entries must not be empty")
        self.entries = list(entries)
        self.k1 = k1
        self.b = b
        self.vi_lexicon = vi_lexicon or DEFAULT_VI_LEXICON
        self._doc_tokens = [tokenize(entry.text) for entry in self.entries]
        self._doc_lengths = [len(tokens) for tokens in self._doc_tokens]
        self._avg_length = sum(self._doc_lengths) / len(self._doc_lengths)
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()
        for doc_index, tokens in enumerate(self._doc_tokens):
            frequencies = Counter(tokens)
            for term, frequency in frequencies.items():
                self._postings[term].append((doc_index, frequency))
                document_frequency[term] += 1
        n_docs = len(self.entries)
        self._idf = {
            term: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def _bm25_scores(self, query_tokens: Iterable[str]) -> tuple[dict[int, float], int]:
        scores: dict[int, float] = defaultdict(float)
        matched_terms = 0
        for term in dict.fromkeys(query_tokens):
            postings = self._postings.get(term)
            if not postings:
                continue
            matched_terms += 1
            idf = self._idf[term]
            for doc_index, frequency in postings:
                length_ratio = self._doc_lengths[doc_index] / self._avg_length
                denominator = frequency + self.k1 * (1.0 - self.b + self.b * length_ratio)
                scores[doc_index] += idf * frequency * (self.k1 + 1.0) / denominator
        return scores, matched_terms

    @staticmethod
    def _chapter_from_code(code: str) -> str | None:
        letter = code[0].upper()
        ranges = (
            ("A", "B", "01"), ("C", "D", "02"), ("D", "D", "03"),
            ("E", "E", "04"), ("F", "F", "05"), ("G", "G", "06"),
            ("H", "H", "07"), ("I", "I", "09"), ("J", "J", "10"),
            ("K", "K", "11"), ("L", "L", "12"), ("M", "M", "13"),
            ("N", "N", "14"), ("O", "O", "15"), ("P", "P", "16"),
            ("Q", "Q", "17"), ("R", "R", "18"), ("S", "T", "19"),
            ("V", "Y", "20"), ("Z", "Z", "21"), ("U", "U", "22"),
        )
        # H00-H59 and H60-H95 need the numeric component to separate eye/ear.
        if letter == "H":
            number = int(code[1:3])
            return "07" if number <= 59 else "08"
        # D00-D48 are neoplasms; D50-D89 are blood/immune disorders.
        if letter == "D":
            number = int(code[1:3])
            return "02" if number <= 48 else "03"
        for start, end, chapter in ranges:
            if start <= letter <= end:
                return chapter
        return None

    def route(self, query_vi: str, query_en: str = "", top_k: int = 3) -> RoutingResult:
        english_tokens = tokenize(query_en)
        scores, matched_terms = self._bm25_scores(english_tokens)
        chapter_scores: dict[str, list[float]] = defaultdict(list)
        for doc_index, score in scores.items():
            chapter_scores[self.entries[doc_index].chapter].append(score)

        aggregated: dict[str, float] = {}
        for chapter, values in chapter_scores.items():
            values.sort(reverse=True)
            aggregated[chapter] = values[0] + 0.20 * sum(values[1:3])

        normalized_vi = f" {normalize_text(query_vi)} "
        for chapter, phrases in self.vi_lexicon.items():
            hits = sum(1 for phrase in phrases if f" {normalize_text(phrase)} " in normalized_vi)
            if hits:
                aggregated[chapter] = aggregated.get(chapter, 0.0) + 2.25 * hits

        direct_code_hit = False
        for match in ICD_CODE_RE.finditer(f"{query_vi} {query_en}"):
            chapter = self._chapter_from_code(match.group(0))
            if chapter:
                direct_code_hit = True
                aggregated[chapter] = aggregated.get(chapter, 0.0) + 12.0

        ranked = sorted(aggregated.items(), key=lambda item: (-item[1], item[0]))
        ranked = ranked[: max(1, top_k)]
        if not ranked:
            return RoutingResult((), "low", True, 0.0, matched_terms)

        top_score = ranked[0][1]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = max(0.0, (top_score - second_score) / max(top_score, 1e-9))
        if direct_code_hit or (top_score >= 10.0 and margin >= 0.20 and matched_terms >= 2):
            confidence = "high"
        elif top_score >= 4.0 and margin >= 0.08:
            confidence = "medium"
        else:
            confidence = "low"
        return RoutingResult(
            routes=tuple(ChapterRoute(chapter, float(score)) for chapter, score in ranked),
            confidence=confidence,
            unknown=confidence == "low",
            margin=float(margin),
            matched_terms=matched_terms,
        )
