import requests
import time
import json
import logging
import sys
import os
from typing import List, Dict, Optional

# Add project root to sys.path so we can import src.config when running this script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import settings

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PubMedClient:
    """Client tương tác với NCBI E-utilities và PubTator API để lấy tài liệu y khoa tiếng Anh."""
    
    def __init__(self):
        self.email = settings.NCBI_EMAIL
        self.api_key = settings.NCBI_API_KEY
        self.esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.pubtator_url = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocjson"
        
    def search_pmids(self, query: str, max_results: int = 100) -> List[str]:
        """Tìm kiếm PubMed dựa trên query và trả về danh sách PMID."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "email": self.email
        }
        if self.api_key:
            params["api_key"] = self.api_key
            
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; R2AI-Bot/1.0)'}
        try:
            response = requests.get(self.esearch_url, params=params, headers=headers)
            response.raise_for_status()
            
            # Xử lý trường hợp NCBI trả về trang HTML "Access Denied" (dù HTTP status = 200) thay vì JSON
            if 'application/json' not in response.headers.get('Content-Type', ''):
                logging.error(f"NCBI API từ chối trả về JSON. Có thể IP của bạn đã bị block tạm thời (Access Denied). Phản hồi thực tế: {response.text[:150]}...")
                return []
                
            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            logging.info(f"Tìm thấy {len(pmids)} PMIDs cho truy vấn: '{query}'")
            return pmids
        except Exception as e:
            logging.error(f"Lỗi khi tìm kiếm PMID cho truy vấn '{query}': {e}")
            return []

    def fetch_pubtator_data(self, pmids: List[str]) -> List[Dict]:
        """Tải nội dung bài báo và các annotations y khoa từ PubTator dựa trên danh sách PMID."""
        if not pmids:
            return []
            
        results = []
        # PubTator khuyến nghị chia nhỏ các request (chunking) nếu danh sách PMID quá dài
        chunk_size = 100
        for i in range(0, len(pmids), chunk_size):
            chunk = pmids[i:i + chunk_size]
            params = {
                "pmids": ",".join(chunk)
            }
            try:
                response = requests.get(self.pubtator_url, params=params)
                response.raise_for_status()
                # PubTator API trả về định dạng JSON Lines (nhiều object JSON phân cách bằng dấu xuống dòng)
                for line in response.text.strip().split('\n'):
                    if line:
                        results.append(json.loads(line))
                # Tránh bị khoá API do vượt quá rate limit (thường là 3 request/giây không có API key)
                time.sleep(0.4) 
            except Exception as e:
                logging.error(f"Lỗi khi tải dữ liệu từ PubTator: {e}")
                
        logging.info(f"Đã tải thành công {len(results)} bài báo từ PubTator.")
        return results

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    # Test thử script (dựa trên ví dụ của BTC)
    client = PubMedClient()
    sample_query = "kidney stone urinary obstruction"
    pmids = client.search_pmids(sample_query, max_results=5)
    docs = client.fetch_pubtator_data(pmids)
    
    print("\n--- KẾT QUẢ TEST ---")
    for doc in docs:
        pmid = doc.get('id', 'Unknown')
        # Lấy Text từ passages (Pubtator cấu trúc nội dung trong mảng passages)
        title = "No Title"
        for passage in doc.get('passages', []):
            if passage.get('infons', {}).get('type') == 'title':
                title = passage.get('text', 'No Title')
                break
        print(f"PMID: {pmid} | Tiêu đề: {title[:70]}...")
