#!/bin/bash
# Nén LZMA trên Kaggle CPU (máy local OOM khi nén): bash scripts/kaggle_zip.sh <tag> <out.zip> <deflate1.zip> [deflate2.zip ...]
# → dataset kakaoffical/r2ai-zipjob (thay toàn bộ file) → kernel r2ai-zipjob-run → tải về → chọn bản LỚN NHẤT < 99.5MB → <out.zip>
set -e -o pipefail
cd /home/phong/Desktop/hsk/test2; TAG=$1; OUT=$2; shift 2
# R2AI_ZIP_OWNER=daophong123 (cần KAGGLE_API_TOKEN) để chạy song song trên tài khoản thứ 2
OW=${R2AI_ZIP_OWNER:-kakaoffical}
if [ $OW = kakaoffical ]; then K="env -u KAGGLE_API_TOKEN .venv/bin/kaggle"; D=data/cache/kg_zipjob; KD=kaggle_zipjob; else K=.venv/bin/kaggle; D=data/cache/kg_zipjob_d; KD=kaggle_zipjob_d; fi
mkdir -p $D; echo "{\"title\": \"r2ai-zipjob\", \"id\": \"$OW/r2ai-zipjob\", \"licenses\": [{\"name\": \"CC0-1.0\"}]}" > $D/dataset-metadata.json
if [ "$1" != "--skip-push" ]; then
  rm -f $D/*.zip; for f in "$@"; do ln -f "$f" $D/; done; cp scripts/zip_lzma.py $D/
  $K datasets version -p $D -m "$TAG" 2>&1 | tail -1 | grep -v "not found" || $K datasets create -p $D 2>&1 | tail -1; sleep 90
  until $K datasets status $OW/r2ai-zipjob 2>/dev/null | grep -q ready; do sleep 30; done
  $K kernels push -p $KD 2>&1 | tail -1; touch data/cache/tmp/state/${TAG}_pushed.done; sleep 120
fi
until s=$($K kernels status $OW/r2ai-zipjob-run 2>&1); echo "$s" | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 120; done; echo "$s"
O=data/cache/zipjob_out/$TAG; rm -rf $O; mkdir -p $O
for i in 1 2 3; do $K kernels output $OW/r2ai-zipjob-run -p $O -o 2>&1 | tail -1; [ -s $O/sizes.txt ] && break; sleep 60; done
cat $O/sizes.txt
best=$(awk '$2 < 99500000 {print $2, $1}' $O/sizes.txt | sort -n | tail -1 | cut -d' ' -f2)
[ -n "$best" ] || { echo "không có bản nào < 99.5MB"; exit 1; }
.venv/bin/python -c "import zipfile,sys; z=zipfile.ZipFile('$O/$best'); assert z.namelist()==['submission.json'], z.namelist(); print('zip ok', z.infolist()[0].compress_type)"
cp $O/$best $OUT; echo "chọn $best -> $OUT ($(stat -c %s $OUT))"
