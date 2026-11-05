"""Domain-specific extractors (selectolax CSS rules) for the biggest hosts of the corpus.

Each extractor gets (tree, url) and returns (title, text) or None (-> generic fallback).
Most sites are described declaratively with `Rule`; Q&A sites have custom functions that keep
the question and every doctor answer.

Selectors were derived from real pages (Oct 2026) of: 120ask.com, ask.39.net, cnkang.com,
familydoctor.com.cn, a-hospital.com, zhongyibaodian.net, zydcd.com, wujue.com, youlai.cn,
jbk.39.net, *.39.net articles, msdmanuals.cn, nhathuoclongchau.com.vn, vinmec.com, ...
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .generic import html_to_paras, trafilatura_extract
from .text import join_paras, norm_para

_REGISTRY: list[tuple[str, object]] = []


def register(fn, *hosts):
    for h in hosts:
        _REGISTRY.append((h.lower(), fn))
    return fn


def site(*hosts):
    def deco(fn):
        return register(fn, *hosts)
    return deco


def find_site_extractor(host: str):
    host = host.lower()
    best = None
    for suffix, fn in _REGISTRY:  # longest matching suffix wins
        if host == suffix or host.endswith("." + suffix):
            if best is None or len(suffix) > len(best[0]):
                best = (suffix, fn)
    return best[1] if best else None


def _first(tree, sels):
    for s in sels:
        n = tree.css_first(s)
        if n is not None:
            return n
    return None


def _text(node) -> str:
    return norm_para(node.text(separator=" ")) if node is not None else ""


def _paras(node, remove=()) -> list[str]:
    for sel in remove:
        for n in node.css(sel):
            n.decompose()
    return html_to_paras(node)


def _filter(paras, drop=(), start_after=None, stop_at=None) -> list[str]:
    out = []
    started = start_after is None
    for p in paras:
        q = norm_para(p)
        if not q:
            continue
        if not started:
            if start_after.search(q):
                started = True
            continue
        if stop_at is not None and stop_at.search(q):
            break
        if any(d.search(q) for d in drop):
            continue
        out.append(q)
    if not started:  # marker never seen -> keep everything
        return _filter(paras, drop, None, stop_at)
    return out


@dataclass
class Rule:
    """Declarative site rule.
    content: CSS selectors; all matches of all selectors are concatenated in document order of the
             selector list (first selector's matches first). Nested duplicates are skipped.
    first_only: use only the first selector that matches anything.
    post: if no content selector matches, post-filter trafilatura output instead (drop/start/stop).
    """
    name: str
    content: list = field(default_factory=list)
    title: list = field(default_factory=lambda: ["h1"])
    remove: list = field(default_factory=list)
    drop: list = field(default_factory=list)
    start_after: str | None = None
    stop_at: str | None = None
    first_only: bool = True
    title_strip: str | None = None
    min_chars: int = 1
    strict: bool = False   # content selectors found nothing -> page has no article (no generic fallback)

    def __post_init__(self):
        self._drop = [re.compile(d) for d in self.drop]
        self._start = re.compile(self.start_after) if self.start_after else None
        self._stop = re.compile(self.stop_at) if self.stop_at else None
        self._tstrip = re.compile(self.title_strip) if self.title_strip else None
        self.__name__ = self.name

    def __call__(self, tree, url):
        tnode = _first(tree, self.title)
        title = _text(tnode)
        if not title:
            from .generic import get_title
            title = get_title(tree)
        if self._tstrip:
            title = self._tstrip.sub("", title).strip()
        paras: list[str] = []
        seen = set()
        for sel in self.content:
            nodes = tree.css(sel)
            nodes = [n for n in nodes if not _inside(n, seen) and not _contains(n, seen)]
            for n in nodes:
                seen.add(n.mem_id)
                paras += _paras(n, self.remove)
            if nodes and self.first_only:
                break
        if not paras:
            if self.strict:
                return title, ""
            if not self.content:
                # rule acts as a post-filter on the generic extraction
                txt = trafilatura_extract(tree.html or "", url)
                paras = txt.split("\n\n") if txt else []
            if not paras:
                return None
        paras = _dedupe(_filter(paras, self._drop, self._start, self._stop))
        # drop a leading paragraph identical to the title
        if paras and title and norm_para(paras[0]) == title:
            paras = paras[1:]
        text = join_paras(paras)
        if len(text) < self.min_chars:
            return None
        return title, text


def _dedupe(paras, min_len: int = 15) -> list[str]:
    """Drop repeated paragraphs (pages that embed the article twice, e.g. desktop + mobile copy)."""
    seen, out = set(), []
    for p in paras:
        if len(p) >= min_len:
            if p in seen:
                continue
            seen.add(p)
        out.append(p)
    return out


def _inside(node, ids) -> bool:
    p = node.parent
    while p is not None:
        if p.mem_id in ids:
            return True
        p = p.parent
    return False


def _contains(node, ids) -> bool:
    if not ids:
        return False
    for d in node.traverse(include_text=False):
        if d.mem_id in ids and d.mem_id != node.mem_id:
            return True
    return False


def rule(*hosts, **kw):
    r = Rule(name=kw.pop("name", hosts[0].split(".")[0]), **kw)
    register(r, *hosts)
    return r


# ============================================================================ Q&A sites

_ANS_BOILER = [re.compile(p) for p in (
    r"^以上是对“.*”这个问题的建议", r"^希望(以上|我的)回答.*(帮助|参考)", r"^我要投诉$", r"^举报$",
    r"^微信扫一扫", r"^查看更多关于", r"^(医生回答|提问者追问)[:：]?$",
)]


def _clean_answer(paras):
    out = []
    for p in paras:
        q = norm_para(p)
        if not q or any(r.search(q) for r in _ANS_BOILER):
            continue
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}( \d{2}:\d{2}(:\d{2})?)?", q):
            continue
        out.append(q)
    return out


_TAG_TXT = re.compile(r"</?[A-Za-z][A-Za-z0-9]{0,8}(\s[^<>]{0,80})?/?>")


@site("120ask.com")
def ask120(tree, url):
    """www.120ask.com/question/*.htm : question box + every doctor answer (病情分析/指导意见)."""
    title = _text(_first(tree, [".b_askti h1", "h1#d_askH1", "h1"]))
    paras: list[str] = []
    q = tree.css_first(".b_askcont")
    if q is not None:
        paras += html_to_paras(q)
    for ans in tree.css(".b_answerli"):
        body = ans.css_first(".crazy_new") or ans.css_first(".b_anscont_cont")
        if body is not None:
            # very old answers contain literally escaped markup ("<P>...</P>")
            paras += _clean_answer(_TAG_TXT.sub("", p) for p in html_to_paras(body))
        for add in ans.css(".b_ansaddli"):  # follow-ups: keep only doctor replies
            t = norm_para(add.text(separator=" "))
            m = re.search(r"医生回答[:：]\s*(.+)$", t)
            if m:
                paras.append(m.group(1))
    if not paras:
        return None
    return title, join_paras(paras)


@site("ask.39.net")
def ask39(tree, url):
    """ask.39.net/question/*.html : title, description, all answers (精选回答 + 其他回答)."""
    title = _text(_first(tree, ["h1.ask_tit", ".ask_cont h1", "h1"]))
    paras: list[str] = []
    q = tree.css_first(".ask_hid") or tree.css_first(".ask_cont .txt_ms")
    if q is not None:
        paras += html_to_paras(q)
    answers = tree.css(".sele_all")
    for a in answers:
        for sel in (".doctor_all", ".doc_t_strip", ".doc_hr", ".txt_yuedu", ".zwAll"):
            for n in a.css(sel):
                n.decompose()
        paras += _clean_answer(html_to_paras(a))
    if not answers:
        return None
    return title, join_paras(paras)


@site("ask.familydoctor.com.cn")
def ask_fd(tree, url):
    title = _text(_first(tree, [".question-title h1", "h1"]))
    paras: list[str] = []
    for sel in (".question-cont", ".illness-desc", ".question .cont"):
        for n in tree.css(sel):
            paras += html_to_paras(n)
    for n in tree.css(".answer-cont, .answer .cont, .answer-item .cont"):
        paras += _clean_answer(html_to_paras(n))
    if not paras:
        return None
    return title, join_paras(paras)


# ============================================================================ article / wiki sites

_REL = r"^(相关(文章|阅读|推荐|问题|疾病|药品)|延伸阅读|本文延伸阅读|热门(推荐|文章|阅读)|推荐阅读|猜你喜欢)[:：]?\s*$"

rule("cnkang.com", name="cnkang",
     content=[".detailc", "#endText", ".content_txt", ".article-content"],
     title=[".detail h1", ".detail h2", "#container2 h1", "h1"],
     remove=[".detailb", ".page", ".pages", "script"],
     drop=[r"^核心提示[:：]"], stop_at=_REL)

rule("familydoctor.com.cn", "familydoctor.cn", name="familydoctor",
     content=["#viewContent", ".viewContent", ".article-content", ".endContent"],
     title=[".article-titile h1", ".main h1", "h1"],
     remove=[".adExtra", ".editorshare", ".tiwencont", ".adBox", ".hotyy", ".part-offer", ".page"],
     drop=[r"^（责任编辑", r"^分享到"], stop_at=_REL)

_MW_REMOVE = ["#jump-to-nav", ".mw-jump", "#siteSub", "#contentSub", "#toc", ".toc", "#catlinks",
              ".printfooter", "#fromlink", ".editsection", ".mw-editsection", ".navbox", "table.nav",
              ".noprint", "#mw-js-message", ".mw-headline-anchor", "#bread", ".bread", ".breadcrumb",
              "#adtop", ".ad", "#comment", ".comment", ".reference-text a"]
_MW_DROP = [r"^跳转到[:：]", r"^(A\+医学百科|中医宝典) ?>>", r"^(A\+医学百科|中医宝典) >", r"^目前暂无留言",
            r"^出自(A\+医学百科|中医宝典)", r"^转载请保留", r"^更多.{1,12}$", r"^【中医宝典App】", r"^添加留言$"]
rule("a-hospital.com", name="a_hospital",
     content=["#mw-content-text", "#bodyContent"], title=["h1#firstHeading", "h1"],
     remove=_MW_REMOVE, drop=_MW_DROP,
     stop_at=r"^(参看|参考资料|市场上的.+|更多医学百科条目|相关文献)[:：]?$|^关于“.+”的留言")
rule("zhongyibaodian.net", name="zhongyibaodian",
     content=["#mw-content-text", "#bodyContent", "#content"], title=["h1#firstHeading", "h1"],
     remove=_MW_REMOVE, drop=_MW_DROP,
     stop_at=r"^(参看|参考资料|更多中医宝典条目|相关文献)[:：]?$|^关于“.+”的留言")

rule("zydcd.com", name="zydcd",
     content=[".panel-danger .panel-body", ".panel-primary .panel-body", ".panel-info .panel-body",
              ".panel-default .panel-body", ".col-md-9 .panel:not(.panel-success) .panel-body"],
     title=[".panel-heading h1", "h1", ".panel-heading"], strict=True,
     remove=[".breadcrumb"], title_strip=r"^(中医|中药方|中药|方剂|中医书籍)[:：]")

rule("wujue.com", name="wujue",
     content=[".art-text", ".art-picbox", "#art-text", "#article-main_left .content"],
     title=["#article-main_left h1", ".art-title", "h1"], strict=True,
     drop=[r"^www\.wujue\.com$", r"^文章来源[:：]", r"^更新时间[:：]"])

rule("youlai.cn", name="youlai",
     content=['div[class*="textContent--"]', 'div[class*="articleContent"]', ".article-content",
              ".text-content", 'div[class*="text-content"]'],
     title=['div[class*="mainContent--"] h1', "h1"], first_only=True)

rule("jbk.39.net", name="jbk39",
     content=[".list_left .disease_box:first-child", ".disease_box .information_box", "article.w740",
              ".art-box", ".list_left .article", ".jbk-article", ".list_left"],
     title=[".disease h1", ".list_left h1", ".disease_title h1", "h1"],
     remove=[".disease_type_slick", ".doctor_item", ".drug_ul", ".drug_box", ".more", ".link", "ul.artlink",
             "dl.catalog", ".zz_info", ".label .more"],
     drop=[r"^(标签|目录)[:：]", r"^详细>>", r"^\.*\[详细\]"],
     stop_at=r"^(其他类似疾病|相关文章|疾病用药|医院医生|相关检查|相关症状)\s*(更多)?$")

# all *.39.net article channels (baby, care, fitness, woman, food, fk, cancer, nk, ...): /a/ pages
rule("39.net", name="net39_article",
     content=["#contentText", ".art_con", ".art_content", ".article_content"],
     title=[".art_box h1", ".art_tit h1", "h1"],
     remove=[".art_page", ".pages", ".art_info", ".art_share", ".art_zhuanti", ".hotpic", "#paper"],
     drop=[r"^指导专家[:：]$"], stop_at=_REL)

rule("msdmanuals.cn", name="msd",
     content=['div[class*="TopicMainContent"]', 'section[class*="topic"]', "main"],
     title=["h1"],
     remove=['[class*="Topic_reviewer"]', '[class*="TopicHead_authors"]', '[class*="authors"]',
             'nav', '[class*="TopicNavigation"]', '[class*="Breadcrumb"]', '[class*="SectionTabs"]',
             '[class*="Footer"]', 'footer', '[class*="Sidebar"]'],
     drop=[r"^(完整评审|作者[:：]|同行评审人|审核|本页内容|主题资源|已审核/修订)"],
     title_strip=r"\s*-\s*.*《默沙东诊疗手册.*》$|\s*-\s*MSD诊疗手册.*$")

rule("bingli.iiyi.com", "article.iiyi.com", name="iiyi",
     content=[".detail_content", ".article-content", ".show_content", ".content", "#content"],
     title=["h1"])


rule("qihuangzhishu.com", name="qihuang", content=[],
     drop=[r"电子书", r"^下载(所有)?", r"^【岐黄之术】", r"^(上一页|目录|相关|下一页)$"])

# ============================================================================ Vietnamese sites

_VI_REL = r"^(Xem thêm|Tin liên quan|Bài liên quan|Đọc thêm|Có thể bạn quan tâm|Video đang được quan tâm|Mời độc giả xem thêm)[:：]?\s*"

rule("nhathuoclongchau.com.vn", "tiemchunglongchau.com.vn", name="longchau",
     # sapo box + article body; the generic extractor sometimes returns only the author bio
     content=[".mt-5.mb-6 .rounded-xl p", '[data-theme-element="article"]'], first_only=False, title=["h1"],
     start_after=r"^Lớn hơn$",
     drop=[r"^Mặc định$", r"^Lớn hơn$", r"^\d{2}/\d{2}/\d{4}$", r"^Xem thêm\s*:", r"^Nguồn\s*:"],
     stop_at=r"^(Nguồn tham khảo|Các bài viết liên quan|Xem thêm các bài viết)\b|^Xem thêm\s*:?$")

rule("vinmec.com", name="vinmec", content=[], title=["h1"],
     drop=[r"^Để đặt lịch khám tại viện", r"^Tải và đặt lịch khám tự động", r"^Đăng ký khám",
           r"^XEM THÊM:?$", r"^Nguồn tham khảo:"],
     stop_at=r"^(Để đặt lịch khám tại viện|Bài viết liên quan)")

rule("suckhoedoisong.vn", name="skds", content=[".detail-sapo", '[data-role="content"]',
                                                  ".detail-content", ".detail__content"],
     first_only=False, title=["h1.detail-title", "h1"],
     remove=[".VCSortableInPreviewMode[type=RelatedNewsBox]", ".VCSortableInPreviewMode[type=RelatedOneNews]",
             ".relate-news", ".box-related", ".link-content-footer", "figure", ".PhotoCMS_Caption"],
     drop=[r"^Mời (độc giả|bạn đọc) xem thêm", r" \| .+ \| "], stop_at=_VI_REL)

rule("suckhoecongdongonline.vn", name="skcd", content=[], title=["h1"],
     drop=[r"^\| \|", r"^Video đang được quan tâm"],
     title_strip=r"\s*\|\s*Sức Khỏe Cộng Đồng\s*$", stop_at=_VI_REL)

rule("laodong.vn", name="laodong", content=[".chappeau", ".art-content", ".article-content", "#gallery-ctt"],
     first_only=False, title=["h1.title", "h1"], remove=[".box-related", ".rel-news", "figure figcaption"],
     stop_at=_VI_REL)
