#!/usr/bin/env python3
"""
Genel tutarlılık testi - çeşitli sorular
"""

from rag_system import RAGRetriever
from ollama_rag import OllamaRAGQA
import time

def test_general_consistency():
    """Çeşitli konularda tutarlılık testini çalıştır"""
    
    print("🧪 Genel Tutarlılık Testi Başlatılıyor...")
    
    # Retriever'ı başlat
    retriever = RAGRetriever()
    
    try:
        retriever.load_from_files("faiss_index.bin", "documents.pkl")
        qa_system = OllamaRAGQA(retriever, model_name="gemma3")
        
        # Çeşitli test soruları
        test_questions = [
            "What is Geodi?",
            "Geodi nedir?",
            "How to install Geodi?",
            "What are the system requirements?",
            "Which databases are supported?",
            "API authentication methods"
        ]
        
        print("📋 Test Soruları:")
        for i, q in enumerate(test_questions, 1):
            print(f"  {i}. {q}")
        
        print("\n" + "="*60)
        
        # Her soruyu test et
        for question in test_questions:
            print(f"\n🔍 TEST: {question}")
            print("-" * 40)
            
            try:
                result = qa_system.answer_question(question, top_k=3)
                
                print(f"💬 Cevap: {result['answer'][:200]}...")
                print(f"📊 Güven: {result['confidence']:.3f}")
                print(f"🔧 Method: {result['method']}")
                
                if result['sources']:
                    print(f"📚 Kaynak sayısı: {len(result['sources'])}")
                
            except Exception as e:
                print(f"❌ Hata: {e}")
            
            time.sleep(0.5)  # Kısa bekleme
            
        print("\n" + "="*60)
        print("✅ Test tamamlandı!")
        
    except FileNotFoundError:
        print("❌ Index dosyaları bulunamadı.")
    except Exception as e:
        print(f"❌ Hata: {e}")

if __name__ == "__main__":
    test_general_consistency()
