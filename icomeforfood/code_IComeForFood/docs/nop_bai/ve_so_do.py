"""Vẽ sơ đồ luồng xử lý cho tài liệu thuyết minh (docs/nop_bai/so_do_he_thong.png).   python docs/nop_bai/ve_so_do.py"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BOXES = {  # key: (x, y, w, h, text, colour)
    "url": (0.3, 9.0, 3.2, 1.0, "1.220.598 URL của BTC\n(VI 740K · EN 240K · ZH 240K)", "#e8eef7"),
    "crawl": (0.3, 7.4, 3.2, 1.2, "Thu thập HTML\ncrawl nhà · crawl4ai/trình duyệt\nCommon Crawl · Kaggle · Europe PMC", "#e8eef7"),
    "clean": (0.3, 5.8, 3.2, 1.2, "Trích & làm sạch (v26)\nkhối nội dung chính, metadata\npostfilter + cổng chất lượng", "#e8eef7"),
    "docs": (0.3, 4.4, 3.2, 1.0, "1.216.297 doc (99,65%)", "#d9ead3"),
    "chunk": (0.3, 2.9, 3.2, 1.0, "Chia chunk s400\n3.134.891 chunk", "#d9ead3"),
    "query": (4.3, 9.0, 3.4, 1.0, "1.200 câu hỏi tiếng Việt", "#fff2cc"),
    "trans": (4.3, 7.6, 3.4, 1.0, "Dịch EN/ZH + thuật ngữ\nQwen3.5-9B (llama.cpp)", "#fff2cc"),
    "cand": (4.3, 5.6, 3.4, 1.6, "Ứng viên (chunk)\nBM25 VI/EN/ZH (bm25s, jieba)\nBGE-M3 dense\nBM25 nguyên doc", "#fce5cd"),
    "qemb": (4.3, 3.6, 3.4, 1.6, "Ứng viên (doc)\nQwen3-Embedding-0.6B và -8B\ntrên tiêu đề + đầu văn bản\n(top 500 / câu)", "#fce5cd"),
    "rr": (8.5, 7.0, 3.6, 1.6, "Rerank chunk\nbge-reranker-v2-m3\nđến 4 chunk/doc, câu dịch cho ZH/EN\nđiểm doc = max, ZH/EN −3", "#f4cccc"),
    "fuse": (8.5, 4.8, 3.6, 1.8, "Xếp lại top 300 doc\nz(bge) + z(PRF chủ đề)\n+ 2·z(Qwen3-Emb-8B)\n+ z(Qwen3-Reranker-8B)", "#f4cccc"),
    "out": (8.5, 2.6, 3.6, 1.8, "File nộp\ndoc: cắt theo điểm ≥ −2,5 (100–400)\nchunk: chunk tốt nhất ±1400 ký tự\ncho 185 doc đầu · zip LZMA", "#d0e0e3"),
}
EDGES = [("url", "crawl"), ("crawl", "clean"), ("clean", "docs"), ("docs", "chunk"), ("query", "trans"), ("trans", "cand"),
         ("chunk", "cand"), ("docs", "qemb"), ("cand", "rr"), ("qemb", "rr"), ("rr", "fuse"),
         ("qemb", "fuse"), ("fuse", "out")]

fig, ax = plt.subplots(figsize=(13, 8.6))
ax.set_xlim(0, 12.4); ax.set_ylim(2.3, 10.3); ax.axis("off")
for k, (x, y, w, h, t, c) in BOXES.items():
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="#555", lw=1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9.5)


def side(k, other):
    x, y, w, h = BOXES[k][:4]; ox, oy, ow, oh = BOXES[other][:4]
    if abs((ox + ow / 2) - (x + w / 2)) < 1:  # same column: bottom -> top
        return (x + w / 2, y) if oy < y else (x + w / 2, y + h)
    return (x + w, y + h / 2) if ox > x else (x, y + h / 2)


for a, b in EDGES:
    ax.add_patch(FancyArrowPatch(side(a, b), side(b, a), arrowstyle="-|>", mutation_scale=12, color="#333", lw=1,
                                 connectionstyle="arc3,rad=0.0"))
ax.set_title("Luồng xử lý: từ câu hỏi đến danh sách doc và chunk trả về", fontsize=12)
plt.tight_layout()
plt.savefig(Path(__file__).with_name("so_do_he_thong.png"), dpi=150)
print("saved")
