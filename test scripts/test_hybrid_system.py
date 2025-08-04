#!/usr/bin/env python3
"""
Yeni hibrit sistem testi - HER ZAMAN LLM kullanır
"""

from rag_system import RAGRetriever
from ollama_rag import OllamaRAGQA
import time

def test_hybrid_system():
    """Hibrit sistemi test et"""
    
    print("🧪 Hibrit RAG-LLM Sistem Testi")
    print("HER ZAMAN LLM kullanır, HAM VERİ DÖNDÜRMEZ")
    
    # Retriever'ı başlat
    retriever = RAGRetriever()
    
    try:
        retriever.load_from_files("faiss_index.bin", "documents.pkl")
        qa_system = OllamaRAGQA(retriever, model_name="gemma3")
        
        # Test soruları - farklı türler
        test_questions = [
            "GEODI ile ilgili sorulara cevap verebilir misin?",  # Genel soru + teknik terim
            "Merhaba, nasılsın?",  # Tamamen genel sohbet
            "What is Geodi?",  # Spesifik RAG sorusu
            "Which port does gde-agent use?",  # Önceki problemli soru
            "Bugün hava nasıl?",  # RAG'da olmayan konu
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
                
                print(f"💬 Cevap: {result['answer'][:300]}...")
                print(f"📊 Güven: {result['confidence']:.3f}")
                print(f"🔧 Method: {result['method']}")
                
                # Ham veri kontrolü
                if any(marker in result['answer'] for marker in ['📌 Tutarlı bilgi bulundu:', '🔍 En ilgili bilgi:']):
                    print("❌ PROBLEM: HAM VERİ DÖNDÜRÜLDÜ!")
                else:
                    print("✅ LLM İŞLEDİ")
                
            except Exception as e:
                print(f"❌ Hata: {e}")
            
            time.sleep(0.5)
            
        print("\n" + "="*60)
        print("✅ Test tamamlandı!")
        
    except FileNotFoundError:
        print("❌ Index dosyaları bulunamadı.")
    except Exception as e:
        print(f"❌ Hata: {e}")

if __name__ == "__main__":
    test_hybrid_system()
