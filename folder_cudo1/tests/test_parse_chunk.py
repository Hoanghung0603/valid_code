from medret.chunk.packer import ChunkConfig, chunk_document
from medret.parse.clean import Block, CleanConfig, clean_blocks, learn_templates, normalize_block, related_box_end
from medret.parse.document import assemble
from medret.parse.html import RawBlock, _blocks_from_minimal_html, decode_html, flag_main


def _count(texts):  # 1 token per character: deterministic, offline
    return [len(t) for t in texts]


def test_decode_gbk_page_with_utf8_header():
    raw = '<meta charset="gb2312"><p>中午用茶下</p>'.encode("gbk")
    html, cs = decode_html(raw, "text/html; charset=utf-8")
    assert cs == "gb18030" and "中午用茶下" in html


def test_minimal_html_blocks_keep_link_text_and_ratio():
    blocks = _blocks_from_minimal_html('<h1>标题</h1>\n<p>中午用<a href="/x">茶</a>下</p>\n<a href="/">首页</a>')
    assert [(b.kind, b.text) for b in blocks] == [("h1", "标题"), ("p", "中午用茶下"), ("text", "首页")]
    assert blocks[1].link_ratio == 1 / 5 and blocks[2].link_ratio == 1.0


def test_flag_main_tolerates_altered_main_text():
    full = [RawBlock("text", "首页"), RawBlock("p", "【用法用量】 每服30丸，中午用茶下，晚上用白汤下。"),
            RawBlock("text", "关于我们")]
    main = [RawBlock("p", "【用法用量】 每服30丸，中午用下，晚上用白汤下。")]  # main_content dropped the link "茶"
    flag_main(full, main)
    assert [b.in_main for b in full] == [False, True, False]


def test_normalize_is_whitespace_and_nfc_only():
    assert normalize_block("  Bé  trai ​3\n\n 头痛　怎么办 ") == "Bé trai 3\n头痛 怎么办"


def test_related_box_inside_article_keeps_following_body():
    out = [
        Block("p", "Những người mắc tiểu đường có nguy cơ bị đục thủy tinh thể cao gấp 2 lần người thường."),
        Block("h4", "Tin liên quan"),
        Block("text", "Gọt hàm xong thì đi cấp cứu", 1.0),
        Block("p", "Đục thủy tinh thể là một trong những biến chứng thường thấy của tiểu đường, theo nghiên cứu."),
        Block("p", "Kết quả cho thấy cứ trung bình 1.000 người bị tiểu đường thì có 20,4 người bị đục thủy tinh thể."),
    ]
    assert related_box_end(out, 1) == 3


def test_related_box_at_page_end_eats_teasers():
    teaser = "Nổi cục u ở cổ họng có thể khiến nhiều người lo lắng, nhất là khi khối u tồn tại kéo dài."
    out = [Block("p", "Thân bài."), Block("h2", "Tin cùng chuyên mục"), Block("text", "Thứ Tư, 16/9"),
           Block("h3", "Nổi cục u ở cổ họng?", 1.0), Block("text", teaser), Block("text", "Thứ Năm, 10/9"),
           Block("h3", "U màng não lành tính?", 1.0), Block("text", teaser)]
    assert related_box_end(out, 1) == len(out)


def test_clean_safe_trim_keeps_body_before_late_h1():
    body = "Cùng một sự kiện, có người xem là thử thách, kẻ khác lại thấy bế tắc. Sự khác biệt nằm ở cách tiếp nhận."
    blocks = [RawBlock("text", "Trang chủ", 1.0), RawBlock("p", body), RawBlock("h1", "Nguyên nhân gây stress")]
    kept, _ = clean_blocks(blocks, "Nguyên nhân gây stress", None, CleanConfig())
    assert any(b.text == body for b in kept)


def test_clean_templates_and_teasers():
    pages = [[RawBlock("text", "Theo dõi chúng tôi"), RawBlock("p", f"Bài {i} nói về bệnh.")] for i in range(10)]
    tpl = learn_templates(pages)
    blocks = [RawBlock("h1", "Tiêu đề"), RawBlock("p", "Nội dung chính của bài viết, khá dài để là văn xuôi."),
              RawBlock("text", "Theo dõi chúng tôi"), RawBlock("p", "清热祛湿，行气逐瘀、通淋排石。适用于...[说明书]")]
    kept, dropped = clean_blocks(blocks, "Tiêu đề", tpl, CleanConfig())
    assert [b.text for b in kept] == ["Tiêu đề", "Nội dung chính của bài viết, khá dài để là văn xuôi."]
    assert {r for r, _ in dropped} == {"template", "teaser"}


def test_garbage_lines_are_dropped():
    binary = "".join(chr(c) for c in (0xFFFD, 3, 1, 0xFFFD, 0x7F)) + "ab" + chr(0xFFFD) * 4
    blocks = [RawBlock("h1", "Ngưng thở khi ngủ"), RawBlock("p", "Ngưng thở khi ngủ là một rối loạn giấc ngủ phổ biến."),
              RawBlock("text", binary), RawBlock("text", "u5RxcMOC4bq+4bumw7nhur7DgsO1w7p64buKw4zh" * 3)]
    kept, dropped = clean_blocks(blocks, "Ngưng thở khi ngủ", None, CleanConfig())
    assert len(kept) == 2 and [r for r, _ in dropped] == ["garbage", "garbage"]


def test_repeats_breadcrumbs_and_link_lists():
    sapo = "(TNO) Một nhóm các nhà khoa học Mexico đang nghiên cứu loại vắc xin có thể giúp chống nghiện heroin."
    blocks = [
        RawBlock("text", "有问必答网 → 妇产科 → 计划生育 → 避孕 → 吃紧急避孕药会有什么副作用，有什么事吗"),
        RawBlock("h1", "Đột phá vắc xin"),
        RawBlock("p", "Đột phá vắc xin"),  # title printed again as a paragraph
        RawBlock("p", sapo), RawBlock("p", sapo),
        RawBlock("p", "Vắc xin A: Lần 2: Ít nhất 1 tháng sau lần 1."), RawBlock("p", "Lần 2: Ít nhất 1 tháng sau lần 1."),
        RawBlock("p", "Vắc xin B:"), RawBlock("p", "Lần 2: Ít nhất 1 tháng sau lần 1."),
        RawBlock("text", "重庆男科医院 郑州医大生殖医院 南京儿童 济南癫痫 广州肿瘤 贵阳精神 贵阳皮肤 成都白癜风", 1.0),
    ]
    kept, dropped = clean_blocks(blocks, "Đột phá vắc xin", None, CleanConfig())
    texts = [b.text for b in kept]
    assert texts[0] == "Đột phá vắc xin" and texts.count(sapo) == 1  # breadcrumb trimmed before <h1>
    assert texts.count("Lần 2: Ít nhất 1 tháng sau lần 1.") == 2  # short list lines are structure
    assert {r for r, _ in dropped} == {"before_h1", "repeat", "short_link"}


def test_unit_heading_path_is_the_context_at_its_start():
    blocks = [Block("h1", "T"), Block("p", "a" * 40), Block("h2", "S1"), Block("p", "b" * 30)]
    text, spans = assemble(blocks)
    units = chunk_document(text, spans, _count, ChunkConfig(target_tokens=200, max_tokens=60, min_tokens=5))
    assert units[0].start == 0 and units[0].heading_path == ""


def test_runs_of_headings_respect_max_tokens():
    blocks = [Block("h1", "T")] + [Block("h6", f"Phim số {i} " + "x" * 20) for i in range(40)]
    text, spans = assemble(blocks)
    units = chunk_document(text, spans, _count, ChunkConfig(target_tokens=100, max_tokens=150, min_tokens=10))
    assert max(u.n_tokens for u in units) <= 150


def test_chunks_are_verbatim_and_keep_small_sections_whole():
    blocks = [Block("h1", "T"), Block("p", "a" * 30), Block("h2", "S1"), Block("p", "b" * 50),
              Block("h2", "S2"), Block("p", "c" * 50)]
    text, spans = assemble(blocks)
    cfg = ChunkConfig(target_tokens=60, max_tokens=80, min_tokens=10)
    units = chunk_document(text, spans, _count, cfg)
    assert units[0].start == 0 and units[-1].end == len(text)
    assert all(a.end <= b.start for a, b in zip(units, units[1:], strict=False))
    for name in ("S1" + "\n" + "b" * 50, "S2" + "\n" + "c" * 50):  # each section lies inside one unit
        start = text.index(name)
        assert any(u.start <= start and start + len(name) <= u.end for u in units)


def test_oversize_block_splits_at_sentences():
    para = "".join(f"第{i}句话说明一些事情。" for i in range(40))
    text, spans = assemble([Block("p", para)])
    units = chunk_document(text, spans, _count, ChunkConfig(target_tokens=100, max_tokens=150, min_tokens=10))
    assert len(units) > 1 and all(text[u.start:u.end].endswith("。") for u in units)
    assert "".join(text[u.start:u.end] for u in units) == text


def test_language_prior_caps_and_thresholds_chinese_docs():
    import pandas as pd

    from medret.select.selector import SelectConfig, choose_docs

    cands = pd.DataFrame({"doc_id": [1, 2, 3, 4, 5], "score": [0.9, 0.8, 0.7, 0.6, 0.05],
                          "lang": ["zh", "vi", "zh", "zh", "vi"]})
    assert choose_docs(cands, SelectConfig(tau_doc=0.1)) == [1, 2, 3, 4]
    assert choose_docs(cands, SelectConfig(tau_doc=0.1, max_zh_docs=1)) == [1, 2]
    assert choose_docs(cands, SelectConfig(tau_doc=0.0, zh_tau=0.75)) == [1, 2, 5]


def test_full_doc_span_cap():
    from medret.select.selector import full_doc_span

    assert full_doc_span(100, 10, 20, None) == (0, 100)
    assert full_doc_span(100, 10, 20, 200) == (0, 100)
    assert full_doc_span(1000, 500, 600, 200) == (450, 650)  # centred on the best chunk
    assert full_doc_span(1000, 0, 50, 200) == (0, 200)  # clamped at the start
    assert full_doc_span(1000, 950, 1000, 200) == (800, 1000)  # clamped at the end
