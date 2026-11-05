import subprocess, time, textwrap
# Crawl ZH v2: ghi shard ra đĩa local (/content/docs_zh) vì Drive FUSE không cho rename file tạm,
# rồi cứ 10 phút chép các shard đã hoàn tất sang Drive. Bỏ 39.net (captcha).
script = textwrap.dedent(r'''
cd /content/r2ai
OUT=/content/drive/MyDrive/r2ai
LOC=/content/docs_zh
mkdir -p $LOC $OUT/docs_zh $OUT/logs
pkill -f "src.crawl" ; sleep 3
run() {
  nohup python -m src.crawl --domains "$2" --per-domain $3 --delay $4 --concurrency 96 \
    --out-dir $LOC >> $OUT/logs/crawl2_$1.log 2>&1 &
}
run articles "a-hospital.com,familydoctor.com.cn,familydoctor.cn,zhongyibaodian.net,zydcd.com,wujue.com,youlai.cn,qihuangzhishu.com,pmphai.com,jb39.com,msdmanuals.cn,baidu.com,people.com.cn,99.com.cn" 12 0.05
run cnkang cnkang.com 16 0.05
run ask120 120ask.com 32 0.05
pkill -f "sync_loop" ; nohup bash -c 'exec -a sync_loop bash -c "while true; do sleep 600; cp -n /content/docs_zh/corpus-*.parquet /content/drive/MyDrive/r2ai/docs_zh/ 2>/dev/null; cp /content/r2ai/data/cache/crawl.sqlite /content/drive/MyDrive/r2ai/crawl_zh.sqlite; done"' >/dev/null 2>&1 &
''')
open('/content/r2ai/colab/restart_zh.sh', 'w').write(script)
print(subprocess.run('bash /content/r2ai/colab/restart_zh.sh', shell=True, capture_output=True, text=True).stderr[-500:])
while True:
    time.sleep(600)
    print(subprocess.run('tail -qn1 /content/drive/MyDrive/r2ai/logs/crawl2_*.log; ls /content/docs_zh | wc -l; ls /content/drive/MyDrive/r2ai/docs_zh | wc -l; df -h /content | tail -1',
                         shell=True, capture_output=True, text=True).stdout, flush=True)
