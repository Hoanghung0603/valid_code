"""Schema, prompt and deterministic validation for R2AI query decomposition."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd


QUERY_TYPES = (
    "definition",
    "diagnosis",
    "lab_interpretation",
    "treatment",
    "medication",
    "safety",
    "risk",
    "comparison",
    "care_guidance",
    "prevention",
    "prognosis",
    "procedure",
    "population_specific",
    "multi_intent",
    "other",
)


def _string_array(description: str) -> dict[str, Any]:
    return {
        "type": "array",
        "description": description,
        "maxItems": 6,
        "items": {"type": "string", "maxLength": 96},
    }


def query_spec_json_schema() -> dict[str, Any]:
    atom = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "atom_id": {"type": "string", "pattern": "^a[1-5]$"},
            "intent": {"type": "string", "maxLength": 96},
            "retrieval_query_vi": {"type": "string", "maxLength": 256},
            "retrieval_query_en": {"type": "string", "maxLength": 256},
            "retrieval_query_zh": {"type": "string", "maxLength": 256},
            "must_match": _string_array("Concepts that evidence must address."),
            "optional_terms": _string_array("Useful synonyms or related medical terms."),
            "exclusions": _string_array("Concepts that would make a candidate irrelevant."),
        },
        "required": [
            "atom_id",
            "intent",
            "retrieval_query_vi",
            "retrieval_query_en",
            "retrieval_query_zh",
            "must_match",
            "optional_terms",
            "exclusions",
        ],
    }
    entities = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "diseases": _string_array("Diseases or diagnoses."),
            "symptoms": _string_array("Symptoms, signs or findings."),
            "interventions": _string_array("Treatments, procedures or behaviors."),
            "medications": _string_array("Medication names or drug classes."),
            "populations": _string_array("Age, sex, pregnancy or other population constraints."),
            "outcomes": _string_array("Requested outcomes, risks or benefits."),
        },
        "required": [
            "diseases",
            "symptoms",
            "interventions",
            "medications",
            "populations",
            "outcomes",
        ],
    }
    synonym_pair = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "casual": {"type": "string", "maxLength": 96},
            "medical": {"type": "string", "maxLength": 96},
            "language": {"type": "string", "enum": ["vi", "en", "zh", "mixed"]},
        },
        "required": ["casual", "medical", "language"],
    }
    subject_fact = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "subject": {"type": "string", "maxLength": 64},
            "fact": {"type": "string", "maxLength": 160},
            "polarity": {"type": "string", "enum": ["affirmed", "negated", "uncertain"]},
        },
        "required": ["subject", "fact", "polarity"],
    }
    cardinality = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "min_docs": {"type": "integer", "minimum": 1, "maximum": 60},
            "max_docs": {"type": "integer", "minimum": 1, "maximum": 60},
            "rationale": {"type": "string", "maxLength": 160},
        },
        "required": ["min_docs", "max_docs", "rationale"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query_id": {"type": "integer"},
            "query_type": {"type": "string", "enum": list(QUERY_TYPES)},
            "canonical_vi": {"type": "string", "maxLength": 512},
            "query_en": {"type": "string", "maxLength": 512},
            "query_zh": {"type": "string", "maxLength": 512},
            "evidence_atoms": {"type": "array", "minItems": 1, "maxItems": 4, "items": atom},
            "entities": entities,
            "subject_facts": {"type": "array", "maxItems": 10, "items": subject_fact},
            "temporal_constraints": _string_array("Time, duration or chronology constraints."),
            "negations": _string_array("Negated concepts that must remain negated."),
            "exclusions": _string_array("Global evidence exclusions."),
            "synonym_pairs": {"type": "array", "maxItems": 8, "items": synonym_pair},
            "needs_multihop": {"type": "boolean"},
            "expected_evidence_cardinality": cardinality,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": [
            "query_id",
            "query_type",
            "canonical_vi",
            "query_en",
            "query_zh",
            "evidence_atoms",
            "entities",
            "subject_facts",
            "temporal_constraints",
            "negations",
            "exclusions",
            "synonym_pairs",
            "needs_multihop",
            "expected_evidence_cardinality",
            "confidence",
        ],
    }


@dataclass(frozen=True)
class QuerySpec:
    query_id: int
    data: dict[str, Any]

    @classmethod
    def validate(cls, value: Mapping[str, Any], expected_query_id: int | None = None) -> list[str]:
        errors: list[str] = []
        required = set(query_spec_json_schema()["required"])
        missing = sorted(required - set(value))
        if missing:
            errors.append(f"missing fields: {', '.join(missing)}")
            return errors
        try:
            query_id = int(value["query_id"])
        except (TypeError, ValueError):
            errors.append("query_id must be an integer")
            query_id = -1
        if expected_query_id is not None and query_id != int(expected_query_id):
            errors.append(f"query_id mismatch: expected {expected_query_id}, got {query_id}")
        if value.get("query_type") not in QUERY_TYPES:
            errors.append("query_type is not allowed")
        for key in ("canonical_vi", "query_en", "query_zh"):
            if not isinstance(value.get(key), str) or not value[key].strip():
                errors.append(f"{key} must be non-empty")
        atoms = value.get("evidence_atoms")
        if not isinstance(atoms, list) or not 1 <= len(atoms) <= 4:
            errors.append("evidence_atoms must contain 1..4 items")
        else:
            atom_ids: list[str] = []
            for index, atom in enumerate(atoms):
                if not isinstance(atom, dict):
                    errors.append(f"evidence_atoms[{index}] must be an object")
                    continue
                atom_id = str(atom.get("atom_id", "")).strip()
                atom_ids.append(atom_id)
                for key in ("atom_id", "intent", "retrieval_query_vi", "retrieval_query_en", "retrieval_query_zh"):
                    if not isinstance(atom.get(key), str) or not atom[key].strip():
                        errors.append(f"evidence_atoms[{index}].{key} must be non-empty")
                for key in ("must_match", "optional_terms", "exclusions"):
                    if not isinstance(atom.get(key), list):
                        errors.append(f"evidence_atoms[{index}].{key} must be an array")
            if any(not atom_id for atom_id in atom_ids) or len(atom_ids) != len(set(atom_ids)):
                errors.append("atom_id values must be unique and non-empty")
        entities = value.get("entities")
        entity_keys = {"diseases", "symptoms", "interventions", "medications", "populations", "outcomes"}
        if not isinstance(entities, dict) or not entity_keys.issubset(entities):
            errors.append("entities is incomplete")
        else:
            for key in entity_keys:
                if not isinstance(entities[key], list):
                    errors.append(f"entities.{key} must be an array")
        facts = value.get("subject_facts")
        if not isinstance(facts, list):
            errors.append("subject_facts must be an array")
        else:
            for index, fact in enumerate(facts):
                if not isinstance(fact, dict):
                    errors.append(f"subject_facts[{index}] must be an object")
                    continue
                if not str(fact.get("subject", "")).strip() or not str(fact.get("fact", "")).strip():
                    errors.append(f"subject_facts[{index}] subject/fact must be non-empty")
                if fact.get("polarity") not in {"affirmed", "negated", "uncertain"}:
                    errors.append(f"subject_facts[{index}].polarity is invalid")
        for key in ("temporal_constraints", "negations", "exclusions", "synonym_pairs"):
            if not isinstance(value.get(key), list):
                errors.append(f"{key} must be an array")
        if not isinstance(value.get("needs_multihop"), bool):
            errors.append("needs_multihop must be boolean")
        cardinality = value.get("expected_evidence_cardinality")
        if not isinstance(cardinality, dict):
            errors.append("expected_evidence_cardinality must be an object")
        else:
            try:
                minimum = int(cardinality["min_docs"])
                maximum = int(cardinality["max_docs"])
                if not 1 <= minimum <= maximum <= 60:
                    errors.append("cardinality must satisfy 1 <= min_docs <= max_docs <= 60")
            except (KeyError, TypeError, ValueError):
                errors.append("cardinality fields are invalid")
        try:
            confidence = float(value["confidence"])
            if not 0 <= confidence <= 1:
                errors.append("confidence must be between 0 and 1")
        except (TypeError, ValueError):
            errors.append("confidence must be numeric")
        return errors


EXPLICIT_NEGATION_RE = re.compile(
    r"\b(?:không|chưa|chẳng)\s+(?:có|bị|mắc|nhiễm|từng)|\bâm\s+tính\b",
    flags=re.IGNORECASE,
)


def _normalize_subject(subject: str) -> str | None:
    text = " ".join(str(subject).casefold().split())
    if not text:
        return None
    if any(token in text for token in ("bác sĩ", "doctor", "physician", "clinician")):
        return None
    if "vợ chồng" in text or "couple" in text:
        return "couple"
    if text in {"self", "tôi", "em", "cháu", "mình", "người hỏi", "patient"}:
        return "self"
    if "wife" in text or "người vợ" in text or text == "vợ":
        return "wife"
    if "husband" in text or "người chồng" in text or text == "chồng":
        return "husband"
    if any(token in text for token in ("thai nhi", "em bé", "bé nhà", "con trai", "con gái", "child")):
        return "child"
    if any(token in text for token in ("bố", "ba", "cha", "father")):
        return "father"
    if any(token in text for token in ("mẹ", "má", "mother")):
        return "mother"
    return text


def validate_query_spec_for_query(
    value: Mapping[str, Any],
    *,
    expected_query_id: int,
    original_query: str,
) -> list[str]:
    errors = QuerySpec.validate(value, expected_query_id=expected_query_id)
    if errors:
        return errors
    has_explicit_negation = bool(EXPLICIT_NEGATION_RE.search(str(original_query)))
    facts = value.get("subject_facts") or []
    preserved_negation = any(
        isinstance(fact, dict) and fact.get("polarity") == "negated" for fact in facts
    )
    if has_explicit_negation and not preserved_negation:
        errors.append(
            "explicit clinical negation must be represented by a subject_fact with polarity=negated"
        )
    clinical_subjects = {
        normalized
        for fact in facts
        if isinstance(fact, dict)
        for normalized in [_normalize_subject(str(fact.get("subject", "")))]
        if normalized is not None
    }
    if has_explicit_negation and len(clinical_subjects) >= 2:
        errors.append(
            "unsafe multi-subject negation: keep the Exp122 fallback unless independently reviewed"
        )
    return errors


def build_query_spec_messages(
    query_id: int,
    query: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    user_payload = {
        "query_id": int(query_id),
        "original_query_vi": str(query).strip(),
    }
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def build_repair_messages(
    query_id: int,
    query: str,
    system_prompt: str,
    raw_output: str,
    validation_errors: Sequence[str],
) -> list[dict[str, str]]:
    payload = {
        "query_id": int(query_id),
        "original_query_vi": str(query).strip(),
        "invalid_output": raw_output[:12000],
        "validation_errors": list(validation_errors),
        "instruction": "Return a corrected QuerySpec only. Preserve medical meaning and negation.",
    }
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _query_features(text: str) -> dict[str, bool | int | str]:
    normalized = " ".join(str(text).lower().split())
    words = re.findall(r"\w+", normalized, flags=re.UNICODE)
    word_count = len(words)
    if word_count <= 10:
        length_bin = "xs"
    elif word_count <= 20:
        length_bin = "s"
    elif word_count <= 40:
        length_bin = "m"
    elif word_count <= 70:
        length_bin = "l"
    else:
        length_bin = "xl"
    patterns = {
        "causal": r"\b(vì sao|tại sao|nguyên nhân|do đâu)\b",
        "recommendation": r"\b(nên|có nên|làm gì|điều trị|uống|dùng)\b",
        "population": r"\b(mang thai|bà bầu|mẹ bầu|trẻ em|trẻ sơ sinh|người già|cao tuổi)\b",
        "comparison": r"\b(so với|khác nhau|tốt hơn|hay là)\b",
        "enumeration": r"[;:]|(?:^|\s)[a-dA-D1-9][\).]",
        "multi_clause": r"[,;].{12,}[?]|\b(và|nhưng|đồng thời|ngoài ra)\b",
    }
    flags = {name: bool(re.search(pattern, normalized)) for name, pattern in patterns.items()}
    priority = ("comparison", "causal", "population", "recommendation", "multi_clause", "enumeration")
    intent = next((name for name in priority if flags[name]), "general")
    return {"word_count": word_count, "length_bin": length_bin, "intent_bin": intent, **flags}


def select_stratified_pilot(queries: pd.DataFrame, n: int = 120, seed: int = 20261023) -> pd.DataFrame:
    if not {"id", "query"}.issubset(queries.columns):
        raise ValueError("query table must contain id and query")
    if queries["id"].duplicated().any():
        raise ValueError("query IDs must be unique")
    frame = queries[["id", "query"]].copy()
    features = frame["query"].map(_query_features).apply(pd.Series)
    frame = pd.concat([frame, features], axis=1)
    frame["stratum"] = frame["length_bin"].astype(str) + "|" + frame["intent_bin"].astype(str)
    frame["sample_key"] = frame["id"].map(
        lambda value: hashlib.sha256(f"{seed}:{int(value)}".encode("utf-8")).hexdigest()
    )
    groups = {
        name: group.sort_values("sample_key").to_dict("records")
        for name, group in frame.groupby("stratum", sort=True)
    }
    selected: list[dict[str, Any]] = []
    while len(selected) < min(n, len(frame)):
        progressed = False
        for name in sorted(groups):
            if groups[name] and len(selected) < n:
                selected.append(groups[name].pop(0))
                progressed = True
        if not progressed:
            break
    result = pd.DataFrame(selected).drop(columns=["sample_key"])
    return result.sort_values("id").reset_index(drop=True)
