#!/bin/bash
# Đóng gói dữ liệu + kết quả trung gian để chia sẻ (Google Drive) cho BTC – xem README §2, §4.
#   bash scripts/package_release.sh      → release/{vi_docs.tar, zh_docs.parquet, artifacts_final.tar, artifacts_features.tar, submission_private_P7_v76b.zip, SHA256SUMS}
set -e
cd "$(dirname "$0")/.."; R=release; mkdir -p $R
# 1) 735k trang VI
[ -s $R/vi_docs.tar ] || tar cf $R/vi_docs.tar -C data/docs .
# 2) 222k trang ZH
cp -u data/cache/kg_zhdocs/zh_docs.parquet $R/zh_docs.parquet
# 3) tái hiện nhanh bài nộp cuối (README §4)
[ -s $R/artifacts_final.tar ] || tar cf $R/artifacts_final.tar \
  data/cache/vi_preds_300.jsonl data/cache/rr_gn_top200.npy data/cache/query_plans.jsonl \
  data/cache/dense_gn_top300_v9.json data/cache/dense_bgem3_chunks_v9.json data/cache/dense_chunks_top_v9.json \
  data/cache/ltrx7_out/ltr_order_oof34.json data/cache/ltrx7_out/ltr_order_oof34_probs.json data/cache/zh_probs7.parquet
# 4) đặc trưng để chạy lại LTR VI / ZH (README §5 bước 13, 17)
[ -s $R/artifacts_features.tar ] || tar cf $R/artifacts_features.tar \
  data/cache/rr_bge_all_v9.parquet data/cache/rr_win_v9.parquet data/cache/ce_ft_v9.parquet data/cache/ce_ftwin_v9.parquet \
  data/cache/gnft_v9.parquet data/cache/ce2b_v9.parquet data/cache/knn_exp.json data/cache/e5_v9 data/cache/dense_gn \
  data/cache/zhfeat3_out/zh_pairs.parquet data/cache/zhfeat3_out/zh_ids.json data/cache/zhfeat3_out/zh_emb.npy \
  data/cache/cezh_v1.parquet data/cache/zh_qfeat.parquet data/cache/qtrans_out/q_zh.json data/cache/zh_qemb_bgem3.npy
# 5) file đã nộp private
cp -u submissions/private/P7_v76b.zip $R/submission_private_P7_v76b.zip
( cd $R && sha256sum vi_docs.tar zh_docs.parquet artifacts_final.tar artifacts_features.tar submission_private_P7_v76b.zip > SHA256SUMS )
ls -la $R; echo "→ tải toàn bộ thư mục $R/ lên Google Drive, bật 'Anyone with the link – Viewer', điền link vào README.md và docs/THUYET_MINH.md"
