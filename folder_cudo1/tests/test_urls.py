from medret.data.urls import canonical_key, host_language, host_of


def test_scheme_www_fragment_slash_are_ignored():
    keys = {
        canonical_key("http://www.120ask.com/question/1000024.htm"),
        canonical_key("https://www.120ask.com/question/1000024.htm"),
        canonical_key("https://120ask.com/question/1000024.htm#comment"),
    }
    assert len(keys) == 1
    assert canonical_key("https://jbk.39.net/nj1/") == canonical_key("https://jbk.39.net/nj1")


def test_percent_encoding_is_decoded():
    a = "https://www.a-hospital.com/w/%E4%B8%87%E6%BA%90%28%E7%A6%8F%E5%B7%9E%29"
    b = "https://www.a-hospital.com/w/%E4%B8%87%E6%BA%90(%E7%A6%8F%E5%B7%9E)"
    assert canonical_key(a) == canonical_key(b)


def test_tracking_params_dropped_but_content_params_kept():
    base = "https://www.baidu.com/bh/dict/ydxx_1"
    assert canonical_key(base + "?from=dicta&sf_ref=med_pc&sf_ch=ch_med_pc") == canonical_key(base)
    p = "https://test.pmphai.com/jeesitede/appcase/toPcDetail?sessionId=&knowledgeLibPrefix=case&id="
    assert canonical_key(p + "1") != canonical_key(p + "2")


def test_host_language():
    assert host_language(host_of("https://vnexpress.net/x")) == "vi"
    assert host_language(host_of("https://bachmai.gov.vn/x")) == "vi"
    assert host_language(host_of("https://www.msdmanuals.cn/x")) == "zh"
    assert host_language(host_of("http://www.cnkang.com/x")) == "zh"
