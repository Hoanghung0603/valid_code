from __future__ import annotations

import pandas as pd

from src.llm.query_spec import QuerySpec, select_stratified_pilot, validate_query_spec_for_query


def valid_spec():
    return {
        "query_id": 7,
        "query_type": "safety",
        "canonical_vi": "Thuốc này có an toàn cho phụ nữ mang thai không?",
        "query_en": "Is this medicine safe during pregnancy?",
        "query_zh": "这种药物在怀孕期间安全吗？",
        "evidence_atoms": [
            {
                "atom_id": "a1",
                "intent": "medication safety in pregnancy",
                "retrieval_query_vi": "an toàn thuốc phụ nữ mang thai",
                "retrieval_query_en": "medication safety during pregnancy",
                "retrieval_query_zh": "妊娠期 药物 安全性",
                "must_match": ["thuốc", "mang thai"],
                "optional_terms": ["thai kỳ"],
                "exclusions": [],
            }
        ],
        "entities": {
            "diseases": [],
            "symptoms": [],
            "interventions": [],
            "medications": ["thuốc"],
            "populations": ["phụ nữ mang thai"],
            "outcomes": ["an toàn"],
        },
        "subject_facts": [
            {"subject": "người hỏi", "fact": "đang mang thai", "polarity": "affirmed"}
        ],
        "temporal_constraints": ["trong thai kỳ"],
        "negations": [],
        "exclusions": [],
        "synonym_pairs": [{"casual": "bầu", "medical": "phụ nữ mang thai", "language": "vi"}],
        "needs_multihop": False,
        "expected_evidence_cardinality": {"min_docs": 3, "max_docs": 20, "rationale": "focused"},
        "confidence": 0.9,
    }


def test_valid_query_spec_passes():
    assert QuerySpec.validate(valid_spec(), expected_query_id=7) == []


def test_query_spec_catches_id_and_cardinality_errors():
    value = valid_spec()
    value["expected_evidence_cardinality"]["min_docs"] = 30
    value["expected_evidence_cardinality"]["max_docs"] = 10
    errors = QuerySpec.validate(value, expected_query_id=8)
    assert any("query_id mismatch" in error for error in errors)
    assert any("cardinality" in error for error in errors)


def test_semantic_guard_requires_explicit_negation_to_be_preserved():
    value = valid_spec()
    errors = validate_query_spec_for_query(
        value,
        expected_query_id=7,
        original_query="Vợ tôi không có virus viêm gan B",
    )
    assert any("negation" in error for error in errors)
    value["subject_facts"] = [
        {"subject": "vợ", "fact": "nhiễm viêm gan B", "polarity": "negated"}
    ]
    errors = validate_query_spec_for_query(
        value,
        expected_query_id=7,
        original_query="Vợ không có virus viêm gan B",
    )
    assert errors == []


def test_multi_subject_negation_uses_safe_fallback():
    value = valid_spec()
    value["subject_facts"].append(
        {"subject": "vợ", "fact": "nhiễm viêm gan B", "polarity": "negated"}
    )
    errors = validate_query_spec_for_query(
        value,
        expected_query_id=7,
        original_query="Tôi nhiễm HBV nhưng vợ tôi không có virus viêm gan B và đang mang thai",
    )
    assert any("unsafe multi-subject negation" in error for error in errors)


def test_couple_and_family_possessives_do_not_create_false_multiple_subjects():
    value = valid_spec()
    value["subject_facts"] = [
        {"subject": "vợ chồng", "fact": "mắc bệnh rõ ràng", "polarity": "negated"}
    ]
    assert validate_query_spec_for_query(
        value,
        expected_query_id=7,
        original_query="Vợ chồng không mắc bệnh rõ ràng nhưng chưa có con",
    ) == []

    value["subject_facts"] = [
        {"subject": "bố em", "fact": "di căn ung thư", "polarity": "negated"},
        {"subject": "bác sĩ", "fact": "khuyên theo dõi", "polarity": "affirmed"},
    ]
    assert validate_query_spec_for_query(
        value,
        expected_query_id=7,
        original_query="Bố em không có biểu hiện di căn, bác sĩ khuyên theo dõi",
    ) == []


def test_nonclinical_process_negation_does_not_trigger_guard():
    value = valid_spec()
    assert validate_query_spec_for_query(
        value,
        expected_query_id=7,
        original_query="Phôi không nuôi được ngày 5 và chuyển phôi cũng không được",
    ) == []


def test_stratified_pilot_is_deterministic_and_unique():
    queries = pd.DataFrame(
        {
            "id": range(1, 51),
            "query": [
                ("Vì sao tôi đau đầu và nên làm gì? " if i % 3 == 0 else "Thuốc này dùng thế nào? ")
                + ("phụ nữ mang thai " if i % 5 == 0 else "")
                + str(i)
                for i in range(1, 51)
            ],
        }
    )
    first = select_stratified_pilot(queries, n=20, seed=42)
    second = select_stratified_pilot(queries, n=20, seed=42)
    assert first["id"].tolist() == second["id"].tolist()
    assert len(first) == 20
    assert not first["id"].duplicated().any()
