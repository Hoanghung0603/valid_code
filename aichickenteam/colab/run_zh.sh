#!/bin/bash
# Crawl tiếng Trung trên Colab. Gọi: !bash colab/run_zh.sh   (sau khi đã mount Drive)
# Kết quả: /content/drive/MyDrive/r2ai/docs_zh/corpus-*.parquet ; log + state sao lưu ở /content/drive/MyDrive/r2ai/
set -e
cd /content/r2ai
OUT=/content/drive/MyDrive/r2ai
mkdir -p $OUT/docs_zh $OUT/logs data/cache data/ViBioMIR

pip -q install curl_cffi==0.16.3 trafilatura==2.2.0 selectolax==0.4.13 lxml_html_clean polars pyarrow zstandard orjson charset-normalizer pyyaml huggingface_hub >/dev/null
[ -f data/ViBioMIR/links_corpus.parquet ] || hf download AIGuruTinix/ViBioMIR links_corpus.parquet --repo-type dataset --local-dir data/ViBioMIR >/dev/null

# khôi phục state nếu phiên trước đã chạy
[ -f $OUT/crawl_zh.sqlite ] && [ ! -f data/cache/crawl.sqlite ] && cp $OUT/crawl_zh.sqlite data/cache/crawl.sqlite || true
# 39.net: mỗi sub-host (ask/jbk/care/baby) có ngân sách riêng
sed -i 's/^  split_hosts: \[\]/  split_hosts: [39.net]/' configs/default.yaml

run() {  # name, domains, per-domain, delay
  nohup python -m src.crawl --domains "$2" --per-domain $3 --delay $4 --concurrency 96 \
    --out-dir $OUT/docs_zh > $OUT/logs/crawl_$1.log 2>&1 &
}
run articles "a-hospital.com,familydoctor.com.cn,familydoctor.cn,zhongyibaodian.net,zydcd.com,wujue.com,youlai.cn,qihuangzhishu.com,pmphai.com,jb39.com,msdmanuals.cn,baidu.com,people.com.cn,99.com.cn" 12 0.05
run cnkang cnkang.com 16 0.05
run n39 39.net 24 0.05
run ask120 120ask.com 32 0.05

# sao lưu state mỗi 10 phút
nohup bash -c "while true; do sleep 600; sqlite3 data/cache/crawl.sqlite '.backup $OUT/crawl_zh.sqlite' 2>/dev/null || cp data/cache/crawl.sqlite $OUT/crawl_zh.sqlite; done" > /dev/null 2>&1 &
sleep 60
tail -n 2 $OUT/logs/crawl_*.log
