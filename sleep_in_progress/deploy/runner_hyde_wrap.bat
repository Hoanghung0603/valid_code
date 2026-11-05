@echo off
rem hyde_wrap.bat <tag>  -- nap Qwen3.5-9B (LM Studio) -> chay stage hyde cua runner -> gỡ model.
rem Chay qua schtasks (r2ai_hyde). Log: D:\r2ai\data\sub\runner\<tag>\hyde_wrap.log
setlocal
set TAG=%1
set L=D:\r2ai\data\sub\runner\%TAG%\hyde_wrap.log
echo start %DATE% %TIME% > %L%
%USERPROFILE%\.lmstudio\bin\lms.exe load qwen/qwen3.5-9b --gpu max --context-length 4096 --identifier qwen35-9b -y >> %L% 2>&1
if errorlevel 1 goto fail
set RUNNER_ARGS=%RUNNER_ARGS%
call D:\r2ai\runner\runner_b660.bat %TAG% hyde
if errorlevel 1 goto fail
%USERPROFILE%\.lmstudio\bin\lms.exe unload --all >> %L% 2>&1
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
exit /b 0
:fail
%USERPROFILE%\.lmstudio\bin\lms.exe unload --all >> %L% 2>&1
echo HYDE WRAP FAILED %DATE% %TIME% >> %L%
echo exit fail >> %L%
exit /b 1
