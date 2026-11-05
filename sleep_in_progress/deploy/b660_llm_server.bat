@echo off
rem Chay LM Studio server + nap Qwen3.5 9B va giu phien song (tien trinh khoi chay tu SSH bi chet theo phien).
rem Dung: schtasks /End /TN r2ai_llm. Log: D:\r2ai\data\llm\llm.log
if not exist D:\r2ai\data\llm mkdir D:\r2ai\data\llm
%USERPROFILE%\.lmstudio\bin\lms.exe server start --port 1234 > D:\r2ai\data\llm\llm.log 2>&1
%USERPROFILE%\.lmstudio\bin\lms.exe load qwen/qwen3.5-9b --gpu max --context-length 4096 --identifier qwen35-9b -y >> D:\r2ai\data\llm\llm.log 2>&1
echo loaded %DATE% %TIME% >> D:\r2ai\data\llm\llm.log
:loop
ping -n 61 localhost > nul
goto loop
