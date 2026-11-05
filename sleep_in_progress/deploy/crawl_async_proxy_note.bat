@echo off
rem Transfer proxy secrets to b660 private tools file (minimal access).
rem Reads local Webshare_100_proxies.txt (gitignored), uploads FIRST 8 valid
rem lines only to D:\r2ai\data\crawl\tools\proxy_secrets.txt (700-equivalent,
rem own tools dir, never in repo). Existence-only check afterwards.
setlocal
echo [proxy] manual step: run local transfer script, not this bat directly
exit /b 2
