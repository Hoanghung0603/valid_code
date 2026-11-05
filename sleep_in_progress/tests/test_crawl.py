import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from crawl import HostCtl, State, tier_of  # noqa: E402
from extract import classify  # noqa: E402
from shards import ShardWriter, iter_records, read_shard  # noqa: E402


class _FakeContent:
    """Mô phỏng StreamReader trả dữ liệu thành nhiều mảnh nhỏ."""

    def __init__(self, data: bytes, piece: int):
        self.data, self.piece = data, piece

    async def iter_chunked(self, n):
        for i in range(0, len(self.data), self.piece):
            yield self.data[i:i + self.piece]


class _FakeResp:
    def __init__(self, data, piece):
        self.content = _FakeContent(data, piece)


def test_read_capped_reads_all_pieces_and_caps():
    import asyncio
    from crawl import read_capped
    data = b"x" * 200_000
    assert asyncio.run(read_capped(_FakeResp(data, 1500), 3_000_000)) == data  # không bị cắt ở mảnh đầu
    assert len(asyncio.run(read_capped(_FakeResp(data, 1500), 10_000))) == 10_000


def test_classify():
    assert classify(200, "https://a.vn/x", "x" * 300) == "ok"
    assert classify(200, "https://image.39.net/verify.html?referer=x", "滑动拼图验证") == "blocked"
    assert classify(200, "https://a.cn/x", "滑动拼图验证\n加载中…") == "blocked"
    assert classify(403, "", "") == "blocked"
    assert classify(404, "", "") == "permanent"
    assert classify(200, "https://a.vn/x", "ngắn") == "permanent"
    assert classify(None, "", "", "TimeoutError") == "transient"


def test_tier():
    assert tier_of("vinmec.com") == 1 and tier_of("baonghean.vn") == 1
    assert tier_of("120ask.com") == 2
    assert tier_of("zhongyibaodian.net") == 3
    assert tier_of("zysjonline.com") == 4 and tier_of("nhathuoclongchau.com.vn") == 4


def test_hostctl_backoff_and_giveup():
    c = HostCtl("h", delay=1.0, min_delay=1.0)
    c.on_result("blocked", cooldown=99, give_up_after=4)
    assert c.delay == 2.0
    c.on_result("blocked", 99, 4)
    assert c.on_result("blocked", 99, 4) == 99  # 3 lần liên tiếp -> cooldown
    c.on_result("blocked", 99, 4)
    assert c.gave_up
    c2 = HostCtl("h", delay=4.0, min_delay=1.0)
    c2.on_result("ok", 99, 4)
    assert abs(c2.delay - 3.6) < 1e-9 and c2.consec_block == 0


def test_hostctl_gives_up_when_mostly_blocked():
    from crawl import HostCtl
    c = HostCtl("h", delay=1.0, min_delay=1.0)
    # xen kẽ ok/blocked: không bao giờ chặn liên tiếp, nhưng 60% bị chặn -> phải bỏ host
    for i in range(200):
        c.on_result("blocked" if i % 5 in (0, 1, 2) else "ok", cooldown=0, give_up_after=12)
        if c.gave_up:
            break
    assert c.gave_up and c.total <= 60
    ok = HostCtl("h2", delay=1.0, min_delay=1.0)
    for i in range(200):  # 20% bị chặn: giữ host
        ok.on_result("blocked" if i % 5 == 0 else "ok", cooldown=0, give_up_after=12)
    assert not ok.gave_up


def test_bandwidth_limiter():
    from crawl import BandwidthLimiter
    off = BandwidthLimiter(0)
    assert off.reserve(10_000_000, now=0.0) == 0.0
    lim = BandwidthLimiter(8)  # 8 Mbit/s = 1 MB/s
    assert lim.reserve(1_000_000, now=0.0) == 0.0          # request đầu đi ngay
    assert abs(lim.reserve(1_000_000, now=0.0) - 1.0) < 1e-9  # request sau phải chờ 1s
    assert abs(lim.reserve(500_000, now=0.0) - 2.0) < 1e-9
    assert lim.reserve(1, now=100.0) == 0.0                # nhàn rỗi thì không tích nợ


def test_shards_roundtrip_and_truncation(tmp_path):
    w = ShardWriter(tmp_path, "crawl", max_records=250)
    for i in range(600):
        w.write({"id": i, "text": "nội dung 内容"})
    w.close()
    recs = list(iter_records(tmp_path))
    assert [r["id"] for r in recs] == list(range(600))
    assert len(list(tmp_path.glob("crawl-*.jsonl.zst"))) == 3
    # cắt cụt file cuối giữa chừng: vẫn đọc được các frame hoàn chỉnh trước đó
    last = sorted(tmp_path.glob("crawl-*.jsonl.zst"))[0]
    data = last.read_bytes()
    last.write_bytes(data[: len(data) - 7])
    assert len(list(read_shard(last))) >= 200


def test_state_resume(tmp_path):
    s = State(tmp_path)
    s.record(1, "ok"); s.record(2, "blocked"); s.record(3, "permanent"); s.record(2, "transient:TimeoutError")
    s.close()
    s2 = State(tmp_path)
    assert not s2.pending(1) and not s2.pending(3)
    assert s2.pending(2)  # 2 lần thử < MAX_ATTEMPTS
    s2.record(2, "blocked"); s2.close()
    assert not State(tmp_path).pending(2)  # đủ 3 lần
