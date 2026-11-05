import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from scripts.chunk_v2 import select_chunks, paragraphs, get_query_terms, DEFAULT_CFG


def test_substring_verbatim_property():
    text = (
        "Khoa Tiêu hóa Bệnh viện Bạch Mai là đơn vị đầu ngành.\n\n"
        "Viêm dạ dày trợt xung huyết là tình trạng niêm mạc dạ dày bị tổn thương, "
        "xuất hiện các vết trợt đỏ rải rác hoặc tập trung thành đám.\n\n"
        "Triệu chứng thường gặp bao gồm đau thượng vị, đầy bụng, ợ hơi, buồn nôn. "
        "Bệnh nhân cần tuân thủ phác đồ điều trị của bác sĩ chuyên khoa.\n\n"
        "Phòng ngừa tái phát bằng cách ăn uống đúng giờ, hạn chế đồ cay nóng và rượu bia."
    )
    query = "triệu chứng viêm dạ dày trợt xung huyết"
    plan = {
        "keywords": {"vi": ["viêm dạ dày", "trợt xung huyết", "thượng vị"]},
        "vi": ["dấu hiệu viêm dạ dày", "triệu chứng đau dạ dày"],
    }

    chunks = select_chunks(query, plan, text, lang="vi", cfg={"target_chars": 200, "max_chunks": 2})
    assert len(chunks) >= 1
    for c in chunks:
        assert c in text, f"Chunk must be an exact verbatim substring of text: {c!r}"


def test_chinese_chunk_selection():
    text_zh = (
        "患者咨询：高血压患者日常饮食需要注意什么？能吃芹菜和降压茶吗？\n\n"
        "病情分析：高血压是一种常见的慢性心血管疾病。长期高血压可导致脑卒中、冠心病及肾功能衰竭等严重并发症。日常饮食调理对控制血压非常关键。\n\n"
        "指导意见：\n"
        "1. 低盐低脂饮食：每天食盐摄入量应控制在5克以下，减少高脂肪及油炸食品的摄入。\n"
        "2. 多吃富含钾、钙的食物：如新鲜蔬菜（芹菜、菠菜）、新鲜水果及豆制品，有助于促进钠的排出。\n"
        "3. 降压茶可作为辅助，但不能替代正规降压药物。建议按医嘱规律服用降压药并定期监测血压。"
    )
    query = "Cao huyết áp cần ăn uống kiêng cữ gì"
    plan_zh = {
        "zh": ["高血压患者饮食注意事项", "高血压日常饮食调理"],
        "keywords": {"zh": ["高血压", "饮食", "低盐低脂", "降压"]},
    }

    chunks = select_chunks(query, plan_zh, text_zh, lang="zh", cfg={"target_chars": 150, "max_chunks": 1})
    assert len(chunks) == 1
    assert chunks[0] in text_zh
    # Should select the recommendation / advice paragraph
    assert "低盐低脂饮食" in chunks[0] or "病情分析" in chunks[0]


def test_empty_and_short_documents():
    assert select_chunks("", None, "") == []
    assert select_chunks("query", None, "   ") == []

    short = "Một câu duy nhất rất ngắn."
    res = select_chunks("câu", None, short, lang="vi", cfg={"target_chars": 500})
    assert res == [short]
    assert res[0] in short


def test_lead_bonus_prioritization():
    # Two paragraphs with similar token match; first paragraph should get lead bonus
    p1 = "Bệnh sốt xuất huyết là bệnh truyền nhiễm cấp tính do virus Dengue gây ra."
    p2 = "Virus Dengue lây truyền qua vết đốt của muỗi vằn Aedes aegypti nhiễm virus."
    text = f"{p1}\n\n{p2}"

    res = select_chunks("virus Dengue gây sốt xuất huyết", None, text, lang="vi", cfg={"target_chars": 100, "lead_bonus": 0.5})
    assert len(res) == 1
    assert p1 in res[0]


def test_answer_bonus_prioritization():
    q = "Hỏi: Tôi bị đau tức ngực trái kéo dài 3 ngày nay, có nguy hiểm không?"
    filler = "Thông tin giới thiệu về phòng khám đa khoa quốc tế với trang thiết bị hiện đại hàng đầu."
    ans = "Bác sĩ tư vấn: Đau tức ngực trái là dấu hiệu cảnh báo bệnh lý mạch vành nguy hiểm. Bạn cần đi khám ngay."
    text = f"{q}\n\n{filler}\n\n{ans}"

    res = select_chunks(
        "đau tức ngực trái có nguy hiểm không",
        None,
        text,
        lang="vi",
        cfg={"target_chars": 150, "answer_bonus": 0.8, "lead_bonus": 0.0, "bonus_mode": "multiplicative"},
    )
    assert len(res) >= 1
    assert "Bác sĩ tư vấn" in res[0]


def test_multi_chunk_distinctness():
    # Long text with distinct sections
    paras = [
        f"Mục {i}: Nội dung chi tiết về vấn đề y tế số {i} với các thông tin bệnh học quan trọng cần biết."
        for i in range(10)
    ]
    text = "\n\n".join(paras)

    chunks = select_chunks("y tế bệnh học", None, text, lang="vi", cfg={"target_chars": 120, "max_chunks": 2, "dedup_iou_threshold": 0.3})
    assert len(chunks) == 2
    assert chunks[0] in text
    assert chunks[1] in text
    assert chunks[0] != chunks[1]


def test_paragraphs_merging():
    # Headers shorter than min_chars should be merged with following paragraph
    header = "1. Triệu chứng lâm sàng:"
    body = "Người bệnh thường có biểu hiện sốt cao đột ngột từ 39 đến 40 độ C, kèm theo rét run và đau đầu dữ dội."
    text = f"{header}\n\n{body}"

    spans = paragraphs(text, min_chars=50)
    assert len(spans) == 1
    assert text[spans[0][0] : spans[0][1]] == text
