# R2AI Evidence Review UI

UI local phục vụ kiểm toán release Exp143. Mặc định chỉ có **replay cache đã khóa**; không mô phỏng inference mới và không sinh tư vấn/chẩn đoán y khoa.

```powershell
python -m src.release.cli --config configs/release_exp143.json build-cache
python -m pip install -r demo/medical_retrieval/requirements.txt
python -m uvicorn demo.medical_retrieval.app:app --host 127.0.0.1 --port 8000
```

Mở `http://127.0.0.1:8000`. Lần đầu bấm evidence có thể chậm vì resolver tìm `doc_id` trong source snapshot; kết quả exact substring được cache trong RAM của process.

## Knowledge graph tùy chọn

Giao diện có một mục **Experimental** tách khỏi prediction Exp143. Graph chỉ nối alias VI/EN/ZH → khái niệm → tài liệu có đúng thuật ngữ trong text snapshot. Không suy ra quan hệ bệnh–triệu chứng hay hiệu quả điều trị, không tính điểm, không thay output replay.

Nếu có `data/reference/concept_graph_v1.sqlite` và `.manifest.json` khớp `configs/medical_concepts_v1.json`, query khớp từ điển sẽ cho xem tối đa 8 ứng viên corpus, alias và URL. Với query ngoài từ điển hoặc khi chưa cung cấp SQLite, UI tự dùng **replay evidence graph**: chỉ nối thuật ngữ xuất hiện trong query/chunk, hoặc đường truy hồi đã lưu khi không có overlap. Mỗi cạnh ghi rõ cơ sở của nó; không tự sinh thêm quan hệ y khoa. Để rebuild concept sidecar:

```powershell
python scripts/build_concept_graph.py --output data/reference/concept_graph_v1.sqlite
python scripts/audit_concept_graph.py
```

Graph corpus là sidecar ~113 MB, chia sẻ riêng nếu muốn demo cả lớp khái niệm trên máy khác; replay evidence fallback không cần file này. Không đặt SQLite trong source ZIP và không gọi hai lớp visualization này là thành phần đã tạo bài private.
