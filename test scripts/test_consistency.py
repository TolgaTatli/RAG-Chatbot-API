#!/usr/bin/env python3
"""
RAG sistemi tutarlılık testi
Aynı soruyu birden fazla kez sorup sonuçları karşılaştırır
"""

from rag_system import RAGRetriever
from ollama_rag import OllamaRAGQA
import time

def test_consistency():
    """Tutarlılık testini çalıştır"""
    
    print("🧪 RAG Tutarlılık Testi Başlatılıyor...")
    
    # Retriever'ı başlat
    retriever = RAGRetriever()
    
    try:
        retriever.load_from_files("faiss_index.bin", "documents.pkl")
        qa_system = OllamaRAGQA(retriever, model_name="gemma3")
        
        # Test soruları
        test_questions = [
            "Which port does gde-agent use?",
            "gde-agent port",
            "geodi gde-agent port",
            "What port is used by GDE Agent?",
            "GDE discovery agent port number"
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
                
                print(f"💬 Cevap: {result['answer']}")
                print(f"📊 Güven: {result['confidence']:.3f}")
                print(f"🔧 Method: {result['method']}")
                
                if result['sources']:
                    print(f"📚 En iyi kaynak (skor: {result['sources'][0]['score']:.3f}):")
                    print(f"    {result['sources'][0]['text'][:100]}...")
                
            except Exception as e:
                print(f"❌ Hata: {e}")
            
            time.sleep(1)  # API'yi rahatlatmak için
            
        print("\n" + "="*60)
        print("✅ Test tamamlandı!")
        
    except FileNotFoundError:
        print("❌ Index dosyaları bulunamadı. Önce data_processor.py çalıştırın.")
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_consistency()
