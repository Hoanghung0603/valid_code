import json
import re

def compute_f2(precision, recall):
    if precision == 0 and recall == 0:
        return 0.0
    return (5 * precision * recall) / (4 * precision + recall)

def calculate_chunk_overlap(pred_text, gt_text):
    """
    Tính độ phủ (Overlap) giữa 2 chunks bằng Token Recall.
    Nếu số từ giao nhau chiếm > 50% số từ của Ground Truth, coi như đạt chuẩn.
    """
    pred_tokens = set(re.findall(r'\w+', pred_text.lower()))
    gt_tokens = set(re.findall(r'\w+', gt_text.lower()))
    
    if not gt_tokens:
        return 0.0
        
    intersection = pred_tokens.intersection(gt_tokens)
    recall_score = len(intersection) / len(gt_tokens)
    return recall_score

def evaluate(ground_truth_path, prediction_path):
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
        
    with open(prediction_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
        
    # Map id -> data
    gt_dict = {item['id']: item for item in gt_data}
    pred_dict = {item['id']: item for item in pred_data}
    
    doc_f2_scores = []
    chunk_f2_scores = []
    
    for query_id, gt in gt_dict.items():
        pred = pred_dict.get(query_id, {"relevant_docs": [], "relevant_chunks": []})
        
        # 1. DOCUMENT LEVEL EVALUATION
        gt_docs = set(gt.get("relevant_docs", []))
        pred_docs = set(pred.get("relevant_docs", []))
        
        if len(gt_docs) == 0:
            doc_p, doc_r = 0.0, 0.0 # Bỏ qua câu hỏi không có đáp án
        else:
            intersect_docs = gt_docs.intersection(pred_docs)
            doc_p = len(intersect_docs) / len(pred_docs) if len(pred_docs) > 0 else 0.0
            doc_r = len(intersect_docs) / len(gt_docs)
            
        doc_f2 = compute_f2(doc_p, doc_r)
        doc_f2_scores.append(doc_f2)
        
        # 2. CHUNK LEVEL EVALUATION
        gt_chunks = gt.get("relevant_chunks", [])
        pred_chunks = pred.get("relevant_chunks", [])
        
        # Một pred_chunk được tính là relevant nếu overlap > 0.5 với bất kỳ gt_chunk nào
        OVERLAP_THRESHOLD = 0.5
        
        relevant_pred_count = 0
        for p_chunk in pred_chunks:
            # Kiểm tra xem p_chunk có match với bất kỳ gt_chunk nào cùng doc_id không
            is_match = False
            for g_chunk in gt_chunks:
                if p_chunk.get('doc_id') == g_chunk.get('doc_id'):
                    overlap = calculate_chunk_overlap(p_chunk.get('chunk_text', ''), g_chunk.get('chunk_text', ''))
                    if overlap >= OVERLAP_THRESHOLD:
                        is_match = True
                        break
            if is_match:
                relevant_pred_count += 1
                
        # Một gt_chunk được coi là "được bao phủ" nếu có ít nhất 1 pred_chunk overlap đủ với nó
        covered_gt_count = 0
        for g_chunk in gt_chunks:
            is_covered = False
            for p_chunk in pred_chunks:
                if p_chunk.get('doc_id') == g_chunk.get('doc_id'):
                    overlap = calculate_chunk_overlap(p_chunk.get('chunk_text', ''), g_chunk.get('chunk_text', ''))
                    if overlap >= OVERLAP_THRESHOLD:
                        is_covered = True
                        break
            if is_covered:
                covered_gt_count += 1
                
        if len(gt_chunks) == 0:
            chunk_p, chunk_r = 0.0, 0.0
        else:
            chunk_p = relevant_pred_count / len(pred_chunks) if len(pred_chunks) > 0 else 0.0
            chunk_r = covered_gt_count / len(gt_chunks)
            
        chunk_f2 = compute_f2(chunk_p, chunk_r)
        chunk_f2_scores.append(chunk_f2)

    # Tính Macro Average
    macro_doc_f2 = sum(doc_f2_scores) / len(doc_f2_scores) if doc_f2_scores else 0.0
    macro_chunk_f2 = sum(chunk_f2_scores) / len(chunk_f2_scores) if chunk_f2_scores else 0.0
    
    print("="*40)
    print("KẾT QUẢ ĐÁNH GIÁ (MACRO AVERAGE)")
    print("="*40)
    print(f"Tổng số queries: {len(gt_dict)}")
    print(f"Document F2 Score: {macro_doc_f2:.4f}")
    print(f"Chunk F2 Score:    {macro_chunk_f2:.4f}")
    print("="*40)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Sử dụng: python scorer.py <ground_truth.json> <predictions.json>")
    else:
        evaluate(sys.argv[1], sys.argv[2])
