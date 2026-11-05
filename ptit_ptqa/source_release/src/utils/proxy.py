def parse_proxy_string(proxy_str: str) -> str | None:
    """
    Tự động chuẩn hóa chuỗi proxy từ nhà cung cấp thành định dạng chuẩn http://user:pass@ip:port
    Hỗ trợ các định dạng:
      1. IP:PORT:username:password (chuẩn 1 dòng phổ biến của các bên bán proxy)
      2. http://username:password@IP:PORT
      3. IP:PORT (proxy không cần mật khẩu)
    """
    if not proxy_str or not isinstance(proxy_str, str):
        return None
    p = proxy_str.strip()
    if not p:
        return None

    # Nếu đã có scheme chuẩn
    if p.startswith(("http://", "https://", "socks5://", "socks4://")):
        return p

    # Cấu trúc: IP:PORT:username:password
    parts = p.split(":")
    if len(parts) == 4:
        ip, port, user, pwd = parts
        return f"http://{user}:{pwd}@{ip}:{port}"
    elif len(parts) == 2:
        ip, port = parts
        return f"http://{ip}:{port}"

    return f"http://{p}"
