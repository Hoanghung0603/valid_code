import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.icd10_router import ICD10ChapterRouter, ICDEntry


def make_router() -> ICD10ChapterRouter:
    return ICD10ChapterRouter(
        [
            ICDEntry("E11", "04", "Type 2 diabetes mellitus"),
            ICDEntry("I10", "09", "Essential primary hypertension"),
            ICDEntry("J45", "10", "Asthma"),
            ICDEntry("K29", "11", "Gastritis and duodenitis"),
            ICDEntry("O20", "15", "Haemorrhage in early pregnancy"),
        ]
    )


def test_routes_diabetes_to_endocrine_chapter() -> None:
    result = make_router().route("Tôi bị tiểu đường type 2", "I have type 2 diabetes mellitus")
    assert result.routes[0].chapter == "04"


def test_routes_gastritis_to_digestive_chapter() -> None:
    result = make_router().route("Viêm dạ dày", "Gastritis")
    assert result.routes[0].chapter == "11"


def test_keeps_ambiguous_lab_query_unknown() -> None:
    result = make_router().route("Chỉ số xét nghiệm này nghĩa là gì?", "What does this laboratory value mean?")
    assert result.unknown


def test_direct_icd_code_is_a_strong_route_signal() -> None:
    result = make_router().route("Bác sĩ chẩn đoán E11", "Diagnosis E11")
    assert result.routes[0].chapter == "04"
    assert result.confidence == "high"
