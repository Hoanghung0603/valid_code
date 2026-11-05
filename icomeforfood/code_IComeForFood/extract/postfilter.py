"""Shared junk filter applied AFTER extraction (DAC_TA section 4, "loại bỏ hoàn toàn").

Drops blocks that are template junk on every page of a site: layout duplicates, author/source
lines, "related" link lists, product cards, UI button labels, footers, image credits, copyright
notices, asker/doctor profile lines on Q&A sites. It never edits a kept block, so kept text stays
verbatim, and it rebuilds `text` and char offsets. Extractors must not dedup on their own: this
module owns deduplication (see dedup_blocks for which repeats are real content).

Validated on the gold set (eval/score_extraction.py): it must not lower recall on any page.
Every rule change is also checked on real docs by sampling the blocks it newly drops.

Usage:
    from extract.postfilter import filter_doc
    doc = filter_doc(doc)              # doc = {"text", "blocks", ...} in schema v2
    python -I extract/postfilter.py <in_dir> <out_dir>   # filter a folder of <id>.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# exact UI labels / section titles that are never content
UI_LABELS = {
    # vi
    "Xem tất cả", "Kích thước chữ", "Mặc định", "Lớn hơn", "Có thể bạn quan tâm", "Chia sẻ", "Đã kiểm duyệt",
    "Không có từ khóa theo chữ cái đã chọn", "☰ Mục lục", "Tham vấn y khoa:", "Bình luận", "Gửi bình luận",
    "Xem thêm:", "Xem thêm", "Đọc thêm", "Tin liên quan", "TIN LIÊN QUAN", "Bài viết liên quan", "Cùng chuyên mục",
    "Các bài viết liên quan", "NỘI DUNG LIÊN QUAN", "Bài viết có hữu ích với bạn?", "Cần tư vấn từ bác sĩ",
    # zh
    "隐藏详情", "我要咨询", "找医生", "举报", "举报/反馈", "搜索", "文", "没有更多了", "更多", "相关推荐", "医生作品",
    "推荐搜索", "优惠推荐", "您最近浏览的手术", "精彩推荐", "手术图文", "最受欢迎育儿热文", "其他人还在看",
    "相关问题", "相关文章推荐", "向医生提问", "栏目关注：", "分享到：", "疾病百科", "投诉", "其他类似问题",
    "回复追问：", "取消 提交",
    # en
    "Test your KnowledgeTake a Quiz!",
}
# a label that also takes the next block with it (label + value)
PAIR_LABELS = {"作者": 1, "评论": 1, "Đã kiểm duyệt nội dung": 1}
# after these markers, following short blocks are a link list until a long block appears
LIST_MARKERS = {"最受欢迎育儿热文", "其他人还在看", "Có thể bạn quan tâm", "Xem thêm:", "Xem thêm", "Tin liên quan",
                "TIN LIÊN QUAN", "Bài viết liên quan", "Cùng chuyên mục", "相关推荐", "推荐搜索", "精彩推荐", "相关文章推荐",
                "Các bài viết liên quan", "NỘI DUNG LIÊN QUAN", "其他类似问题",
                "HỆ THỐNG BỆNH VIỆN ĐA KHOA TÂM ANH"}   # tamanhhospital address/hotline box at the end of each post
# reference sections are content even though their items look like links
REF_HEADING = re.compile(r"^(nguồn|tài liệu)\s+tham\s+khảo|^tham\s+khảo$|^参考(文献|资料)|^references?$", re.I)

# Layout duplicates (DAC_TA 4): drop a later run of >= DUP_RUN consecutive blocks that repeats an earlier
# run, or a single long block (>= DUP_SINGLE chars) seen before. Short repeats are real content in the gold
# set (title restated as question / first heading, TOC entry vs heading, "Ảnh minh họa." captions).
DUP_RUN, DUP_RUN_CHARS, DUP_SINGLE = 3, 100, 100
QA_SITES = ("ask.39.net", "120ask.com")         # askers repeat themselves; the gold keeps every copy

# Source / author lines: only checked in the last TAIL_BLOCKS blocks of a document, because
# "Theo dõi ...", "Theo y học ..." are ordinary Vietnamese content (checked on 32K real docs).
TAIL_BLOCKS = 4
NOT_SOURCE = re.compile(r"^\(?Theo\s+(dõi|chân|đó|như|y học|Y học|Đông y|quan niệm|nghiên cứu|thống kê|khuyến cáo|"
                        r"hướng dẫn|các|một|nhiều|ý kiến)\b|[:：]$")
TAIL_PATTERNS = [re.compile(p) for p in (
    r"^\(?Theo\s[^.!?]{1,90}\)?$",
    r"^[^\s!?]{1,20}(\s[^\s.!?]{1,20}){0,4}\s\(Theo [^)]{1,60}\)$",   # "T. Linh (Theo Aboluowang)"
    r"^(CTV|PV|BTV)\b[^!?]{0,60}$",                            # "CTV Lương Trâm/VOV.VN (Biên dịch)"
)]
AUTHOR = re.compile(
    r"^([A-ZĐ][A-ZĐ.]{0,5}"                                   # PV, A.T, TTXVN
    r"|[A-ZĐÀ-Ỹ][\wÀ-ỹ.]*(\s[A-ZĐÀ-Ỹ][\wÀ-ỹ.]*){1,3}"         # Doãn Phong, Ngọc Lam, Trần Thị Thu
    r"|[\w.-]+\.(vn|com|net|org)"                               # Baoquangbinh.vn
    r"|Ảnh:\s.{1,30})$")

PATTERNS = [re.compile(p) for p in (
    r"^-?>{1,2}\s",                         # "-> ..." / ">> ..." inline related links
    r"^→\s",
    # dates / bylines glued to dates
    r"^•?\s*\d{1,2}/\d{1,2}/\d{4}(\s+\d{1,2}:\d{2})?$",
    r"^Cập nhật( lần cuối)?:?\s*\d",
    r"(?i)nguồn ưu tiên trên Google",       # "Chọn Bệnh viện ... làm nguồn ưu tiên trên Google tìm kiếm" (tamanh, tuoitre, vnexpress)
    r"^(?!Ngày (tham khảo|truy cập))[^\d\s][^\d]{1,30}\d{1,2}/\d{1,2}/\d{4}$",   # a reference's access date stays
    r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}(:\d{2})?)?$",
    r"^\w?\*{3,}\w?\s+\d{4}-\d{2}-\d{2}",       # ask.39 asker "a******5 2014-03-11 14:16"
    r"^(www\.)?[\w-]+(\.[\w-]+)+\s+\d{4}-\d{1,2}-\d{1,2}",  # "www.cnkang.com 2015-8-8 互联网"
    # VN footer / booking / reviewer boxes. Masthead labels only as "Label: value" or alone:
    # "Giám đốc Bệnh viện K cho biết...", "Giấy phép nhập khẩu ... được gia hạn" are content
    r"^(Cơ quan chủ quản|Giám đốc|Tổng [Bb]iên tập|Phó [Tt]ổng [Bb]iên tập|Giấy phép|Đường dây nóng|"
    r"Trụ sở( chính)?)\s*([:：]|$)",
    r"^Giấy phép\b.{0,40}\d+/GP",          # "Giấy phép hoạt động số 101/GP-BTTTT ..."
    r"^(@ Bản quyền|© |Bản quyền|Thông tin tòa soạn|Liên hệ quảng cáo)",
    r"(Để đặt lịch khám|vui lòng bấm số HOTLINE|Liên hệ hotline \d)",
    r"^(Bác sĩ duyệt bài|Dược sĩ Đại học|Nội dung được kiểm duyệt)",
    r"^(GP thiết lập|Không được sao chép|Người chịu trách nhiệm nội dung)",
    r"^Chủ đề\s*:",                         # tag line "Chủ đề:Gạo lứtCaloGiảm cân" (TV, huyPrime1)
    r"^\d+ lượt (xem|chơi|thích)$",
    r"^/\s?[^\s\d]{1,8}$",                  # "/ Lọ", "/ Hộp" unit next to a price
    r"^(Tác giả|Chuyên khoa|Ngày đăng)\s*:.{0,60}$",   # youmed byline box
    r'^\[[a-z_]+(\s+[a-z_]+="[^"]*")*\]$',  # unrendered WordPress shortcode '[wpcode id="293453"]'
    # zh doctor profile, credits, copyright, wiki chrome
    r"^\S{2,10}\s+(主任医师|副主任医师|主治医师|医师|住院医师)(\s|$)",   # "白云山汉方 医师 广州…"
    r"^疾病百科[•\-]",                       # 39.net / a-hospital nav "疾病百科•出血倾向"
    r"^擅长：",
    r"^图片由.{1,40}提供。?$",
    r"^39健康网\(www\.39\.net\)专稿",
    r"^（以上内容仅授权.{0,40}）$",
    r"^39健康网专业医疗保健信息平台",
    r"^中国领先的健康门户网站",
    r"未经(书面)?授权请勿转载。?$",
    r"^跳转到：",
    r"^文章录入[：:]",                        # wujue.com editor credit "文章录入：管理员 责任编辑：…"
    r"^\d+个分类:",
    r"^发表于\s*\d{4}-\d{1,2}-\d{1,2}",          # 120ask follow-up question timestamp
    r"^\d+条追问$",                           # ask.39 follow-up counter; the follow-up itself ("追问： …") stays
    r"^\S{1,12}医生推荐$",                    # youlai box "眼科医生推荐" after the article
    r"^出自A\+医学百科",
    r"^关于“.{1,60}”的留言",
    r"^如您发现信息有误",
)]
# boilerplate longer than the 200-char limit above; anchored, because older extractors glue a
# whole article into one block that contains it
LONG_PATTERNS = [re.compile(p) for p in (
    r"^Để đặt lịch khám tại .{0,40}Quý khách vui lòng",   # vinmec booking box, 200-250 chars
)]
# app promo box (youmed), matched with whitespace removed because older extractors glue its lines
PROMO = re.compile(r"^(Đặtkhámtiệnlợicùng|Đặtkhámkhôngchờđợi)|^((\+\d+(Bệnhviện|Bácsĩ|Phòngkhám))+|Nhắntinvớibácsĩ|"
                   r"Gọivideovớibácsĩ|Muasảnphẩmytếgiátốt|Lưutrữhồsơytế|Đọctinytếchínhthống|TảiứngdụngYouMed)$")
# ask.39: a row of space-separated keywords is a tag or related-search row. Site-bound, because on
# wiki/TCM sites the same shape is an herb list or a table row.
ZH_TAG_ROW = re.compile(r"^(?=.*[一-鿿])[一-鿿A-Za-z0-9]{1,8}( [一-鿿A-Za-z0-9]{1,8}){2,}$")
TAG_ROW_SITES = ("ask.39.net",)
TERMINAL = re.compile(r"[.。!！;；:：]$")
TEASER_LEAD = re.compile(r"\s(\.\.\.|…)$")   # lead cut mid-sentence: "... thu hồi toàn bộ lô mỹ phẩm mặt ..."
# a product card: heading, a few short lines, then a price ("390.000đ") or "contact us" instead of a price
PRICE = re.compile(r"^(\d{1,3}(\.\d{3})+\s?(đ|₫|VNĐ)|Cần tư vấn từ bác sĩ)$")
# Long Châu HTML saved after the browser ran JS (crawl4ai) has a rating widget and a cookie banner that the
# raw HTML lacks. Only these blocks go, never "everything after": on disease pages the FAQ follows the widget.
JS_WIDGET_SITES = ("nhathuoclongchau.com.vn",)
RATING_HEAD = "Đánh giá độ hữu ích của bài viết"
RATING_LABELS = {"Thất vọng", "Không hữu ích", "Bình thường", "Hữu ích", "Rất hữu ích"}   # only right after the head
COOKIE_BANNER = re.compile(r"^(Cookie Consent|Chúng tôi dùng cookie để cá nhân hóa trải nghiệm.{0,250}|"
                           r"Thông báo sản phẩm, dịch vụ mới, quyền lợi khuyến mại|"
                           r"Nâng cao trải nghiệm người dùng và cá nhân hoá)$")


def is_junk(text: str) -> bool:
    t = text.strip()
    if t in UI_LABELS:
        return True
    if len(t) <= 200 and any(p.search(t) for p in PATTERNS):
        return True
    if len(t) <= 400 and any(p.search(t) for p in LONG_PATTERNS):
        return True
    if len(t) <= 200 and PROMO.search(re.sub(r"\s+", "", t)):
        return True
    return False


def _is_author_tail(blocks: list[dict], i: int) -> bool:
    """Source or author line near the end: 'Theo GDTĐ', 'PV', 'A.T', 'Doãn Phong', 'Baoquangbinh.vn'."""
    b = blocks[i]
    if b.get("type") != "paragraph" or i < len(blocks) - TAIL_BLOCKS:
        return False
    t = b["text"].strip()
    if len(t) > 120 or NOT_SOURCE.search(t):
        return False
    if any(p.search(t) for p in TAIL_PATTERNS):
        return True
    return len(t) <= 30 and not re.search(r"[一-鿿\d()]", t) and bool(AUTHOR.match(t))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def dedup_blocks(blocks: list[dict], site: str = "") -> list[dict]:
    t = [_norm(b.get("text", "")) for b in blocks]
    seen: dict[str, list[int]] = {}
    drop, i = set(), 0
    while i < len(t):
        run = 0                                            # longest earlier run that block i starts repeating
        for j in seen.get(t[i], []):
            k = 0
            while i + k < len(t) and j + k < i and t[j + k] == t[i + k]:
                k += 1
            run = max(run, k)
        if run >= DUP_RUN and sum(map(len, t[i:i + run])) >= DUP_RUN_CHARS:
            drop.update(range(i, i + run))
            i += run
            continue
        adjacent = bool(t[i]) and t[i] in seen and seen[t[i]][-1] == i - 1
        # the title emitted twice (page title + h1): adjacent identical headings go at any length
        if adjacent and blocks[i].get("type") == blocks[i - 1].get("type") == "heading":
            drop.add(i)
        # a long title restated as a later heading stays (gold keeps both); its adjacent copy goes
        elif (not site.endswith(QA_SITES) and len(t[i]) >= DUP_SINGLE and t[i] in seen
                and (blocks[i].get("type") != "heading" or adjacent)):
            drop.add(i)
        seen.setdefault(t[i], []).append(i)
        i += 1
    return [b for k, b in enumerate(blocks) if k not in drop]


def _is_content(b: dict) -> bool:
    t = b.get("text", "").strip()
    return len(t) > 100 or bool(re.search(r"[.。!！]$", t))


def _is_linklike(b: dict) -> bool:
    t = re.sub(r"(\.\.\.|…)$", "", b.get("text", "").strip())     # "Trái na: Ngăn ngừa ... tóc và da..."
    return len(t) <= 120 and not TERMINAL.search(t) and not REF_HEADING.search(t)


def _card_blocks(blocks: list[dict]) -> set[int]:
    """Product cards: the heading before a price line, the short lines between, unit/origin lines after."""
    drop = set()
    for i, b in enumerate(blocks):
        if not PRICE.match(b.get("text", "").strip()):
            continue
        for j in range(i - 1, max(i - 4, 0) - 1, -1):
            if len(blocks[j].get("text", "").strip()) > 30:
                break
            if blocks[j].get("type") == "heading":
                drop.update(range(j, i + 1))
                break
        drop.add(i)
        k = i + 1
        while k < len(blocks) and blocks[k].get("type") != "heading" and len(blocks[k].get("text", "").strip()) <= 10:
            drop.add(k)
            k += 1
    return drop


def _cut_tail(blocks: list[dict]) -> list[dict]:
    """Related-article lists at the end: a source line followed only by links, or >= 4 bare headings
    that read like article titles (short section headings with a lost body are kept), or >= 2
    teaser pairs (title + lead cut off with "...") of an inline related box."""
    k = len(blocks)
    while (k >= 2 and blocks[k - 1].get("type") == blocks[k - 2].get("type") == "paragraph"
           and len(blocks[k - 1].get("text", "").strip()) <= 300 and TEASER_LEAD.search(blocks[k - 1].get("text", "").strip())
           and _is_linklike(blocks[k - 2])):
        k -= 2
    if len(blocks) - k >= 4:
        blocks = blocks[:k]
    n = len(blocks)
    for s in range(3, n - 3):                          # related lists can outnumber a short article's blocks
        t = blocks[s].get("text", "").strip()
        # short and comma-free, so "Theo bác sĩ X, người bệnh cần ..." before a closing list is never cut
        if (blocks[s].get("type") == "paragraph" and len(t) <= 50 and "," not in t and not NOT_SOURCE.search(t)
                and any(p.search(t) for p in TAIL_PATTERNS) and all(map(_is_linklike, blocks[s + 1:]))):
            return blocks[:s]
    k = n
    while k > 1 and blocks[k - 1].get("type") == "heading" and _is_linklike(blocks[k - 1]):
        k -= 1
    tail, body = blocks[k:], sum(b.get("type") != "heading" for b in blocks[:k])
    # body >= 5: a page whose paragraphs were lost keeps its section headings
    if len(tail) >= 4 and body >= 5 and sum(len(b.get("text", "").strip()) for b in tail) / len(tail) >= 45:
        return blocks[:k]
    return blocks


def filter_blocks(blocks: list[dict], site: str = "") -> list[dict]:
    keep, skip_next, in_list, in_rating = [], 0, False, False
    cards = _card_blocks(blocks)
    js_widgets = site.endswith(JS_WIDGET_SITES)
    for i, b in enumerate(blocks):
        t = b.get("text", "").strip()
        if skip_next:
            skip_next -= 1
            continue
        if b.get("type") == "heading" and i == 0:          # never drop the title
            keep.append(b)
            continue
        if i in cards:
            continue
        if js_widgets:
            if t == RATING_HEAD or (in_rating and t in RATING_LABELS):
                in_rating = True
                continue
            in_rating = False
            if COOKIE_BANNER.match(t):
                continue
        if t in PAIR_LABELS:
            skip_next = PAIR_LABELS[t]
            continue
        if t in LIST_MARKERS:
            in_list = True
            continue
        if in_list:
            # the link list ends at a reference heading ("NGUỒN THAM KHẢO" after "Có thể bạn quan tâm"),
            # at real text, or at a heading followed by real text; link titles are often headings themselves
            nxt = blocks[i + 1] if i + 1 < len(blocks) else {}
            if b.get("type") == "heading":
                if not (REF_HEADING.search(t) or _is_content(nxt)):
                    continue
            elif len(t) <= 100 and not re.search(r"[.。]$", t):
                continue                                   # still inside the link list
            in_list = False
        if is_junk(t) or _is_author_tail(blocks, i):
            continue
        if site.endswith(TAG_ROW_SITES) and ZH_TAG_ROW.match(t):
            continue
        keep.append(b)
    return _cut_tail(keep)


def rebuild(blocks: list[dict]) -> tuple[str, list[dict]]:
    text, out, pos = "", [], 0
    for b in blocks:
        if out:
            text += "\n\n"
            pos += 2
        nb = dict(b, char_start=pos, char_end=pos + len(b["text"]))
        out.append(nb)
        text += b["text"]
        pos += len(b["text"])
    return text, out


def filter_doc(doc: dict) -> dict:
    blocks = dedup_blocks(list(doc.get("blocks", [])), doc.get("site") or "")
    for _ in range(3):                 # cutting a related list can expose an author line ("PV") at the new tail
        kept = filter_blocks(blocks, doc.get("site") or "")
        if len(kept) == len(blocks):
            break
        blocks = kept
    text, blocks = rebuild(blocks)
    out = dict(doc, text=text, blocks=blocks)
    if "n_chars" in doc:
        out["n_chars"] = len(text)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        (dst / f.name).write_text(json.dumps(filter_doc(doc), ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
