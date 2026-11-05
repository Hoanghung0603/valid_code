import pandas as pd
import pyarrow.parquet as pq

from crawl.classify import classify, decode_html
from crawl.polite import Shards, backoff_pause, group_interval
from crawl.sets import done_ids, rate_group, remaining, split_sets


def test_rate_group_merges_shared_domain_subdomains():
    assert rate_group("ask.39.net") == rate_group("jbk.39.net") == rate_group("39.net") == "39.net"
    assert rate_group("www.cnkang.com") == "www.cnkang.com"
    assert rate_group("x39.net.vn") == "x39.net.vn"
    assert group_interval("39.net", 0.5) == 1.0 and group_interval("www.cnkang.com", 2.0) == 2.0


def _corpus():
    urls = [f"https://www.cnkang.com/a/{i}.html" for i in range(10)] + \
           [f"https://ask.39.net/question/{i}.html" for i in range(7)] + \
           ["https://zysjonline.com/x/1.html"]
    return pd.DataFrame({"id": range(1, len(urls) + 1), "url": urls})


def test_remaining_drops_done_and_excluded_hosts():
    todo = remaining(_corpus(), done={1, 2, 11}, exclude_hosts={"zysjonline.com"})
    assert set(todo.columns) == {"id", "url", "group"}
    assert not {1, 2, 11, 18} & set(todo.id)
    assert set(todo.group) == {"www.cnkang.com", "39.net"}
    assert remaining(_corpus(), done=set(), only_ids=[3, 4]).id.tolist() == [3, 4]


def test_split_sets_is_a_partition_spread_over_every_site():
    todo = remaining(_corpus(), done=set(), exclude_hosts={"zysjonline.com"})
    sets = split_sets(todo, 3)
    ids = [set(s.id) for s in sets]
    assert set().union(*ids) == set(todo.id) and sum(map(len, ids)) == len(todo)
    for s in sets:  # mỗi bộ có phần của mọi site
        assert set(s.group) == {"www.cnkang.com", "39.net"}
    cn = [s[s.group == "www.cnkang.com"].id for s in sets]
    assert cn[0].max() < cn[1].min() and cn[1].max() < cn[2].min()  # bộ 1 giữ id nhỏ nhất
    assert sorted(len(c) for c in cn) == [3, 3, 4]


def test_done_ids_only_counts_final_labels(tmp_path):
    sh = Shards(tmp_path)
    for i, lab in [(1, "ok"), (2, "http_error:404"), (3, "blocked:http_403"), (4, "net_error:ReadTimeout")]:
        sh.add("www.cnkang.com", dict(id=i, url=f"u{i}", label=lab, status=200, fetched_at=0))
    sh.flush_all()
    files = list(tmp_path.glob("www.cnkang.com/part-*.parquet"))
    assert len(files) == 1 and pq.read_table(files[0]).num_rows == 4
    assert done_ids([tmp_path]) == {1, 2}


def test_backoff_doubles_up_to_cap():
    assert [backoff_pause(k, 30, 1800) for k in (1, 2, 3, 7, 20)] == [30, 60, 120, 1800, 1800]


def test_classify_flags_challenges_but_not_articles():
    challenge = "<html><title>Just a moment...</title><script src='/cdn-cgi/challenge-platform/x'></script></html>"
    assert classify(403, {}, challenge, 20, "https://a.vn/b", "https://a.vn/b")[0] == "blocked:cloudflare_challenge"
    article = "<p>" + "Hướng dẫn nhận mã 验证码 qua SMS. " * 200 + "</p>"
    assert classify(200, {}, article, len(article), "https://a.vn/b", "https://a.vn/b")[0] == "ok"
    assert classify(200, {}, article, len(article), "https://a.vn/", "https://a.vn/b")[0] == "redirect_home"
    gone = "<p>" + "Bài viết không tồn tại. " * 50 + "</p>"
    assert classify(404, {}, gone, len(gone), "https://a.vn/b", "https://a.vn/b")[0] == "http_error:404"
    # trên trang lỗi thì từ khoá chung đáng tin: 404 kèm "验证码" là trang captcha
    assert classify(404, {}, article, len(article), "https://a.vn/b", "https://a.vn/b")[0] == "blocked:generic_captcha"


def test_decode_html_prefers_gbk_meta_over_missing_header():
    raw = '<meta charset="gb2312"><p>感冒发烧怎么办</p>'.encode("gbk")
    html, enc = decode_html(raw, "text/html")
    assert "感冒发烧" in html and enc == "gb18030"
