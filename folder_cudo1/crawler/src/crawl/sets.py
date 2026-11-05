"""Chia các URL còn phải crawl thành N bộ để nhiều máy cùng crawl (mỗi máy một IP, mỗi máy vẫn giữ tốc độ lịch sự)."""
import sqlite3
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from crawl.hosts import host_of

# Domain giới hạn tốc độ chung cho mọi subdomain (39.net: ~20 subdomain, captcha khi dồn 5-10 req/s)
# -> cả domain là một nhóm, khoảng cách tối thiểu (giây) giữa 2 request.
SHARED_DOMAINS = {"39.net": 1.0}
# Nhãn coi là đã xong (không crawl lại); blocked:* và net_error:* sẽ được thử lại ở lần chạy sau.
FINAL = ("ok", "http_error", "redirect_home", "soft_404_or_empty", "js_render", "robots_disallowed")


def rate_group(host: str) -> str:
    """host -> nhóm giới hạn tốc độ: domain dùng chung (SHARED_DOMAINS) hoặc chính host."""
    for d in SHARED_DOMAINS:
        if host == d or host.endswith("." + d):
            return d
    return host


def shard_labels(out: Path) -> pd.DataFrame:
    """(id, label, group) của mọi bản ghi trong <out>/<group>/part-*.parquet. Shard hỏng được đổi đuôi .bad."""
    parts = []
    for f in Path(out).glob("*/part-*.parquet"):
        try:
            parts.append(pq.read_table(f, columns=["id", "label"]).to_pandas().assign(group=f.parent.name))
        except Exception:
            f.rename(f.with_suffix(".bad"))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["id", "label", "group"])


def done_ids(shard_dirs=(), sqlite_db: Path | None = None) -> set[int]:
    """Id đã crawl xong: nhãn cuối trong các thư mục shard, và trang có text trong build/crawl.sqlite (nếu có)."""
    ids: set[int] = set()
    for d in shard_dirs:
        s = shard_labels(Path(d))
        ids.update(s[s.label.astype(str).str.startswith(FINAL)].id.astype(int))
    if sqlite_db and Path(sqlite_db).exists():
        db = sqlite3.connect(sqlite_db)
        ids.update(r[0] for r in db.execute("SELECT id FROM pages WHERE status = 200 AND length(text) > 0"))
    return ids


def remaining(corpus: pd.DataFrame, done: set[int], exclude_hosts=(), only_ids=None) -> pd.DataFrame:
    """corpus (id, url) -> các URL chưa xong (id, url, group), sắp theo nhóm rồi id."""
    c = corpus[["id", "url"]].copy()
    if only_ids is not None:
        c = c[c.id.isin(only_ids)]
    c["host"] = c.url.map(host_of)
    c = c[~c.host.isin(set(exclude_hosts)) & ~c.id.isin(done)]
    c["group"] = c.host.map(rate_group)
    return c[["id", "url", "group"]].sort_values(["group", "id"]).reset_index(drop=True)


def split_sets(todo: pd.DataFrame, n: int) -> list[pd.DataFrame]:
    """Mỗi nhóm cắt thành n khoảng id liên tiếp; bộ k nhận phần thứ k của mọi nhóm, nên máy nào cũng rải request
    lên tất cả các site. Bộ 1 giữ id nhỏ nhất (crawler đi theo id tăng dần, máy đang crawl dở giữ bộ 1)."""
    todo = todo.sort_values(["group", "id"]).reset_index(drop=True)
    rank = todo.groupby("group").cumcount()
    size = todo.groupby("group").id.transform("size")
    k = rank * n // size + 1
    return [todo[k == i].reset_index(drop=True) for i in range(1, n + 1)]
