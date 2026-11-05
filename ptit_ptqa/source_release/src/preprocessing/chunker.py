import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseChunker(ABC):
    """
    Class gốc cho tất cả các thuật toán Chunking.
    Cung cấp các hàm tiện ích đếm từ và tách câu.
    """
    def __init__(self, max_tokens: int = 1024, token_overlap: int = 100):
        self.max_tokens = max_tokens
        self.token_overlap = token_overlap
        self.max_words = int(self.max_tokens * 0.75)
        self.overlap_words = int(self.token_overlap * 0.75)

    def count_words(self, text: str) -> int:
        """Đếm số lượng từ trong text."""
        return len(text.split())

    def split_into_sentences(self, text: str) -> List[str]:
        """Tách văn bản thành các câu dựa trên dấu câu."""
        sentences = re.split(r'(?<=[.!?。！？])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    @abstractmethod
    def chunk_document(self, doc_id: str, text: str, url: str = "") -> List[Dict[str, Any]]:
        pass

class SentenceChunker(BaseChunker):
    """
    Thuật toán cắt văn bản thành các Chunk tĩnh dựa trên Paragraph và Sentence.
    Sử dụng Sliding Window với Overlap.
    """
    def chunk_document(self, doc_id: str, text: str, url: str = "") -> List[Dict[str, Any]]:
        if not text:
            return []

        paragraphs = re.split(r'\n\s*\n', text.strip())
        chunks = []
        current_chunk_words = []
        current_word_count = 0
        
        def add_chunk(words_list):
            chunk_text = ' '.join(words_list).strip()
            if chunk_text:
                chunks.append({
                    'doc_id': doc_id,
                    'chunk_index': len(chunks),
                    'content': chunk_text,
                    'url': url,
                    'token_count': self.count_words(chunk_text)
                })

        for paragraph in paragraphs:
            para_words = paragraph.split()
            para_word_count = len(para_words)

            # Nếu nguyên 1 paragraph nhét vừa thì nhét luôn
            if current_word_count + para_word_count <= self.max_words:
                current_chunk_words.extend(para_words)
                current_word_count += para_word_count
            else:
                # Nếu không vừa, phải cắt nhỏ paragraph ra thành từng câu
                sentences = self.split_into_sentences(paragraph)
                for sentence in sentences:
                    sentence_words = sentence.split()
                    sentence_word_count = len(sentence_words)

                    # Nếu thêm câu này vào bị lố words
                    if current_word_count + sentence_word_count > self.max_words and current_chunk_words:
                        add_chunk(current_chunk_words)
                        overlap_start = max(0, len(current_chunk_words) - self.overlap_words)
                        current_chunk_words = current_chunk_words[overlap_start:]
                        current_word_count = len(current_chunk_words)

                    current_chunk_words.extend(sentence_words)
                    current_word_count += sentence_word_count

        if current_chunk_words:
            add_chunk(current_chunk_words)
            
        return chunks

class ContextualChunker(SentenceChunker):
    """
    Chunker kế thừa SentenceChunker nhưng tự động tiêm thêm bối cảnh (First Paragraph)
    vào đầu mỗi Chunk (Kỹ thuật của Anthropic).
    """
    def chunk_document(self, doc_id: str, text: str, url: str = "") -> List[Dict[str, Any]]:
        if not text:
            return []
            
        # Tìm Global Context
        paragraphs = re.split(r'\n\s*\n', text.strip())
        global_context = ""
        for p in paragraphs:
            if len(p.split()) > 5:  
                global_context = ' '.join(p.split()[:40]) # Lấy 40 từ đầu làm context
                break
                
        # Gọi thuật toán cắt cơ bản của cha
        chunks = super().chunk_document(doc_id, text, url)
        
        # Tiêm ngữ cảnh vào đầu chunk
        for chunk in chunks:
            chunk_text = chunk['content']
            if global_context and global_context not in chunk_text:
                chunk['content'] = f"[Context: {global_context}] {chunk_text}"
                chunk['token_count'] = self.count_words(chunk['content'])
                
        return chunks

class StructureChunker(SentenceChunker):
    """
    Chunker tách theo cấu trúc Markdown (Headings).
    Tự động nhận diện các thẻ Header (#, ##, ###) và tiêm cấu trúc này vào các chunk con
    thuộc Header đó. Nếu text không có Header, fallback về SentenceChunker.
    """
    def chunk_document(self, doc_id: str, text: str, url: str = "") -> List[Dict[str, Any]]:
        if not text:
            return []
            
        chunks = []
        # Tách văn bản theo Markdown Headers
        # Regex: match dòng bắt đầu bằng 1-6 dấu #, sau đó là khoảng trắng và nội dung
        sections = re.split(r'(?m)^(#+\s+.*)\n', text.strip())
        
        # Nếu không có header nào, sections sẽ chỉ có 1 phần tử (là toàn bộ text)
        if len(sections) == 1:
            return super().chunk_document(doc_id, text, url)
            
        # sections[0] có thể là đoạn text trước header đầu tiên (nếu có)
        pre_text = sections[0].strip()
        if pre_text:
            chunks.extend(super().chunk_document(doc_id, pre_text, url))
            
        # Xử lý các section có header
        for i in range(1, len(sections), 2):
            header = sections[i].strip()
            content = sections[i+1].strip() if i+1 < len(sections) else ""
            
            if not content:
                continue
                
            # Dùng SentenceChunker cắt phần content của section này
            section_chunks = super().chunk_document(doc_id, content, url)
            
            # Tiêm Header vào đầu mỗi chunk để giữ lại cấu trúc
            for chunk in section_chunks:
                chunk_text = chunk['content']
                injected_text = f"[{header}] {chunk_text}"
                chunk['content'] = injected_text
                chunk['token_count'] = self.count_words(injected_text)
                chunks.append(chunk)
                
        # Cập nhật lại chunk_index cho đúng thứ tự tổng thể
        for idx, chunk in enumerate(chunks):
            chunk['chunk_index'] = idx
            
        return chunks

class SemanticEmbeddingChunker(BaseChunker):
    """
    Chunker tách theo khoảng cách ngữ nghĩa thực sự (Dùng Embedding Model).
    Nhóm các câu liên tiếp có độ tương đồng Cosine cao lại với nhau.
    """
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B", threshold: float = 0.5, embedding_model=None, **kwargs):
        super().__init__(**kwargs)
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            self.cosine_similarity = cosine_similarity
        except ImportError:
            raise ImportError("Vui lòng cài đặt: pip install sentence-transformers scikit-learn")
            
        self.threshold = threshold
        self.model = embedding_model if embedding_model else SentenceTransformer(model_name)
        # Bắt buộc giới hạn token để chống tràn RAM khi văn bản quá dài
        self.model.max_seq_length = 512

    def chunk_document(self, doc_id: str, text: str, url: str = "") -> List[Dict[str, Any]]:
        if not text:
            return []

        sentences = self.split_into_sentences(text)
        if not sentences:
            return []
            
        if len(sentences) <= 2 or self.count_words(text) <= self.max_words:
            return [{
                'doc_id': doc_id,
                'chunk_index': 0,
                'content': text,
                'url': url,
                'token_count': self.count_words(text)
            }]

        safe_sentences = [str(s)[:2000] for s in sentences]
        embeddings = self.model.encode(safe_sentences, batch_size=8)

        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self.cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            similarities.append(sim)

        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_word_count = self.count_words(sentences[0])

        for i in range(len(similarities)):
            sentence_word_count = self.count_words(sentences[i+1])
            
            # Nếu sim thấp hơn ngưỡng hoặc thêm vào bị quá số từ -> Cắt chunk
            if similarities[i] < self.threshold or current_word_count + sentence_word_count > self.max_words:
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append({
                    'doc_id': doc_id,
                    'chunk_index': len(chunks),
                    'content': chunk_text,
                    'url': url,
                    'token_count': self.count_words(chunk_text)
                })
                current_chunk_sentences = [sentences[i+1]]
                current_word_count = sentence_word_count
            else:
                current_chunk_sentences.append(sentences[i+1])
                current_word_count += sentence_word_count

        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append({
                'doc_id': doc_id,
                'chunk_index': len(chunks),
                'content': chunk_text,
                'url': url,
                'token_count': self.count_words(chunk_text)
            })

        return chunks

class Chunker:
    """
    Factory / Strategy Pattern wrapper cho các Chunker.
    Giúp việc thay đổi thuật toán Chunking dễ dàng chỉ qua 1 tham số 'strategy'.
    """
    def __init__(self, strategy: str = "contextual", max_tokens: int = 1024, token_overlap: int = 100, **kwargs):
        strategies = {
            "sentence": SentenceChunker,
            "contextual": ContextualChunker,
            "structure": StructureChunker,
            "semantic": SemanticEmbeddingChunker
        }
        
        if strategy not in strategies:
            raise ValueError(f"Strategy {strategy} is not supported. Supported: {list(strategies.keys())}")
            
        self.chunker = strategies[strategy](max_tokens=max_tokens, token_overlap=token_overlap, **kwargs)

    def chunk_document(self, doc_id: str, text: str, url: str = "") -> List[Dict[str, Any]]:
        return self.chunker.chunk_document(doc_id, text, url)
