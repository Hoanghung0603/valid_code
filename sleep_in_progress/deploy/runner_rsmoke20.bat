@echo off
rem runner_rsmoke20.bat <stage...>  -- launcher cho tag rsmoke20: smoke 20 cau public, delta 56
rem co dinh, ref = vi34/q8b_v33/dense_v30 de bench + so sanh. Moi stage nhan cung RUNNER_ARGS
rem (manifest/ resume khop nhau). Vi du: runner_rsmoke20.bat hyde  |  runner_rsmoke20.bat build stats
set RUNNER_ARGS=--smoke 20 --delta 56 --ref-sub D:\r2ai\data\sub\vi34\submission_vi34.json --ref-q8b D:\r2ai\data\sub\diag33\q8b_v33.json --ref-dense D:\r2ai\data\sub\diag30\dense_v30_top200.json
call D:\r2ai\runner\runner_b660.bat rsmoke20 %*
