"""Đường dẫn và tham số chung của project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Bản phát hành S11: đọc corpus v2 do `medret download` tải về và ghi HTML thẳng vào data/crawl/pages của repo
# (nơi `medret manifest` đọc). Bản gốc trên nhánh feat/crawl-pipeline dùng data/ViBioMIR/links_corpus.parquet (v1)
# và build/pages.
REPO = ROOT.parent
DATA = REPO / "data" / "raw" / "ViBioMIR"
QUERIES = DATA / "query.parquet"
CORPUS = DATA / "links_corpus_vi_en_zh.parquet"

BUILD = ROOT / "build"  # generated, gitignore
CRAWL_DB = BUILD / "crawl.sqlite"
PAGES = REPO / "data" / "crawl" / "pages"  # HTML thô từ crawl diện rộng (crawl.polite): <nhóm site>/part-*.parquet
CRAWL_SETS = BUILD / "crawl_sets"  # set_N.parquet (id, url, group) chia cho nhiều máy cùng crawl
RUNS = BUILD / "runs"
SUBMISSIONS = BUILD / "submission"  # mỗi lần nộp 1 thư mục: <tên>/<tên>.zip, settings.json, scoring/

HF_REPO = "AIGuruTinix/ViBioMIR"
HF_FILES = ("query.parquet", "links_corpus.parquet")

N_QUERIES = 1200

# Ngưỡng của BTC: chunk đúng nếu LCS_tokens(c, g) / |g| >= 0.4
CHUNK_OVERLAP_THRESHOLD = 0.4

# Chunking mặc định (số từ tách theo khoảng trắng / ký tự CJK)
CHUNK_WORDS = 200
