@echo off
rem runner_rsmoke20_z87.bat <stage...>  -- launcher smoke tren z87 (stage dense), cung run_args
rem voi ben b660 cho manifest khop; ref-dense da copy sang z87 de xac minh tai lap.
set RUNNER_ARGS=--smoke 20 --delta 56 --ref-dense D:\r2ai\data\sub\diag30\dense_v30_top200.json
call D:\r2ai\runner\runner_z87.bat rsmoke20 %*
