import numpy as np
import warnings

# Transformer uyarılarını bastır
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple
import pickle
from data_processor import RAGDataProcessor

class RAGRetriever:
    """RAG sistemi için döküman arama sınıfı"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Sentence transformer model adı
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def load_from_processor(self, processor: RAGDataProcessor):
        """RAGDataProcessor'dan index ve dökümanları yükle"""
        self.index = processor.index
        self.documents = processor.documents
        self.model = processor.model

    def load_from_files(self, index_path: str, documents_path: str):
        """Dosyalardan index ve dökümanları yükle"""
        # FAISS indexi yükle
        self.index = faiss.read_index(index_path)

        # Dökümanları yükle
        with open(documents_path, 'rb') as f:
            self.documents = pickle.load(f)

        print(f"Index ve dökümanlar başarıyla yüklendi. Toplam döküman: {len(self.documents)}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Sorguya en benzer dökümanları bul

        Args:
            query: Arama sorgusu
            top_k: Döndürülecek en benzer döküman sayısı

        Returns:
            En benzer dökümanların listesi
        """
        if self.index is None or not self.documents:
            raise ValueError("Index ve dökümanlar yüklenmemiş!")

        # BGE modeli için sorguya prefix ekle
        prefixed_query = f"Represent this sentence for searching relevant passages: {query}"

        # Sorgu için embedding oluştur
        query_embedding = self.model.encode([prefixed_query], normalize_embeddings=True)

        # FAISS ile arama yap
        scores, indices = self.index.search(query_embedding, top_k)

        # Sonuçları hazırla
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):  # Geçerli index kontrolü
                result = {
                    'rank': i + 1,
                    'score': float(score),
                    'document': self.documents[idx],
                    'text': self.documents[idx]['text']
                }
                results.append(result)

        return results

    def get_context_for_query(self, query: str, top_k: int = 3, max_length: int = 2000) -> str:
        """
        Sorgu için bağlam metni oluştur

        Args:
            query: Sorgu
            top_k: Kullanılacak döküman sayısı
            max_length: Maksimum bağlam uzunluğu

        Returns:
            Bağlam metni
        """
        results = self.search(query, top_k)

        context_parts = []
        current_length = 0

        for result in results:
            text = result['text']

            # Maksimum uzunluğu aşmamak için kontrol et
            if current_length + len(text) > max_length:
                remaining_length = max_length - current_length
                if remaining_length > 100:  # En az 100 karakter varsa ekle
                    text = text[:remaining_length] + "..."
                    context_parts.append(f"Döküman {result['rank']}: {text}")
                break

            context_parts.append(f"Döküman {result['rank']}: {text}")
            current_length += len(text)

        return "\n\n".join(context_parts)

    def get_relevant_chunks(self, query: str, top_k: int = 5, min_score: float = 0.1) -> List[Dict]:
        """
        Sorgu ile ilgili chunk'ları skorlarıyla birlikte getir

        Args:
            query: Arama sorgusu
            top_k: Maksimum döküman sayısı
            min_score: Minimum benzerlik skoru

        Returns:
            İlgili chunk'ların listesi
        """
        results = self.search(query, top_k)

        # Minimum skoru karşılayan sonuçları filtrele
        filtered_results = [r for r in results if r['score'] >= min_score]

        return filtered_results

    def similarity_search_with_threshold(self, query: str, threshold: float = 0.2, max_results: int = 10) -> List[Dict]:
        """
        Belirli bir eşik değeri üzerindeki benzer dökümanları bul

        Args:
            query: Arama sorgusu
            threshold: Minimum benzerlik eşiği
            max_results: Maksimum sonuç sayısı

        Returns:
            Eşik değeri üzerindeki sonuçlar
        """
        # Daha fazla sonuç alıp filtreleyeceğiz
        initial_results = self.search(query, max_results * 2)

        # Eşik değeri üzerindeki sonuçları filtrele
        filtered_results = [
            result for result in initial_results
            if result['score'] >= threshold
        ]

        return filtered_results[:max_results]


class SimpleRAGQA:
    """Basit RAG soru-cevap sistemi"""

    def __init__(self, retriever: RAGRetriever):
        """
        Args:
            retriever: RAG retriever instance
        """
        self.retriever = retriever

    def answer_question(self, question: str, top_k: int = 3) -> Dict:
        """
        Soruya cevap ver

        Args:
            question: Soru
            top_k: Kullanılacak döküman sayısı

        Returns:
            Cevap ve meta bilgiler
        """
        # İlgili dökümanları bul
        search_results = self.retriever.search(question, top_k)

        # Bağlam oluştur
        context = self.retriever.get_context_for_query(question, top_k)

        # Basit cevap formatı (gerçek LLM entegrasyonu için genişletilebilir)
        if not search_results:
            return {
                'question': question,
                'answer': "Üzgünüm, bu soruyla ilgili bilgi bulamadım.",
                'context': "",
                'sources': []
            }

        # En iyi eşleşen dökümanı temel cevap olarak kullan
        best_match = search_results[0]

        return {
            'question': question,
            'answer': f"Bulduğum en ilgili bilgi: {best_match['text'][:500]}...",
            'context': context,
            'sources': search_results,
            'confidence': best_match['score']
        }

    def interactive_qa(self):
        """Etkileşimli soru-cevap modu"""
        print("RAG Soru-Cevap Sistemi")
        print("Çıkmak için 'quit' yazın.")
        print("-" * 50)

        while True:
            question = input("\nSorunuz: ").strip()

            if question.lower() in ['quit', 'çık', 'exit', 'q']:
                print("Görüşmek üzere!")
                break

            if not question:
                print("Lütfen bir soru yazın.")
                continue

            try:
                result = self.answer_question(question)

                print(f"\nCevap: {result['answer']}")
                print(f"Güven skoru: {result.get('confidence', 0):.3f}")
                print(f"Kaynak döküman sayısı: {len(result['sources'])}")

                # Detayları göster (isteğe bağlı)
                show_details = input("\nDetayları görmek ister misiniz? (e/h): ").strip().lower()
                if show_details in ['e', 'evet', 'y', 'yes']:
                    print("\nKaynak dökümanlar:")
                    for source in result['sources']:
                        print(f"- Skor: {source['score']:.3f}")
                        print(f"  Metin: {source['text'][:200]}...")
                        print()

            except Exception as e:
                print(f"Hata oluştu: {e}")

def create_test_retriever():
    """Test için basit bir retriever oluştur"""
    retriever = RAGRetriever()

    # Test dökümanları
    test_docs = [
        {"text": "Python bir programlama dilidir. Nesne yönelimli ve yorumlamalı bir dildir.", "source": "test1"},
        {"text": "Machine Learning yapay zeka dalının bir alt alanıdır. Verilerden öğrenmeye dayanır.", "source": "test2"},
        {"text": "RAG (Retrieval Augmented Generation) bilgi getirme ve üretmeyi birleştirir.", "source": "test3"}
    ]

    # Basit in-memory index oluştur
    embeddings = retriever.model.encode([doc["text"] for doc in test_docs])

    # FAISS index oluştur
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity için normalize edilmiş)

    # Embeddings'leri normalize et ve index'e ekle
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))

    retriever.index = index
    retriever.documents = test_docs

    return retriever


if __name__ == "__main__":
    # Örnek kullanım
    print("RAG sistemi başlatılıyor...")

    # Retriever'ı başlat
    retriever = RAGRetriever()

    try:
        # Önceden oluşturulmuş index'i yükle
        retriever.load_from_files("faiss_index.bin", "documents.pkl")

        # QA sistemi oluştur
        qa_system = SimpleRAGQA(retriever)

        # Etkileşimli mod başlat
        qa_system.interactive_qa()

    except FileNotFoundError:
        print("Index dosyaları bulunamadı. Önce data_processor.py çalıştırın.")
    except Exception as e:
        print(f"Hata: {e}")
        import traceback
        traceback.print_exc()
