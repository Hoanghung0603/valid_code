#!/bin/bash
# Chờ kernel $OW/r2ai-buildzip → tải <name>.zip → chọn bản LỚN NHẤT < 99.5MB có tiền tố <prefix> → nộp (submit_retry, chống trùng).
#   bash scripts/kaggle_build_fetch.sh <prefix> <out.zip>
set -e -o pipefail
cd /home/phong/Desktop/hsk/test2; PRE=$1; OUT=$2; O=data/cache/buildzip_out/$PRE
OW=${R2AI_ZIP_OWNER:-kakaoffical}   # daophong123 cần KAGGLE_API_TOKEN
if [ $OW = kakaoffical ]; then K="env -u KAGGLE_API_TOKEN .venv/bin/kaggle"; else K=.venv/bin/kaggle; fi
until s=$($K kernels status $OW/r2ai-buildzip 2>&1); echo "$s" | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 120; done; echo "$(date +%H:%M) $s"
rm -rf $O; mkdir -p $O
for i in 1 2 3; do $K kernels output $OW/r2ai-buildzip -p $O -o --file-pattern "^($PRE.*\.zip|sizes\.txt|.*\.log)$" 2>&1 | tail -1; [ -s $O/sizes.txt ] && break; sleep 60; done
cat $O/sizes.txt; grep -o '"data":"[^"]*' $O/*.log | grep -E "F2 docs|wrote|lỗi|Error" | tail -6 || true
best=$(awk -v p="$PRE" '$2 < 99500000 && index($1, p) == 1 {print $2, $1}' $O/sizes.txt | sort -n | tail -1 | cut -d' ' -f2)
[ -n "$best" ] || { echo "không có bản < 99.5MB"; exit 1; }
.venv/bin/python -c "import zipfile; z=zipfile.ZipFile('$O/$best'); assert z.namelist()==['submission.json']; print('zip ok')"
cp $O/$best $OUT; echo "chọn $best -> $OUT"
bash scripts/submit_retry.sh $OUT 6
