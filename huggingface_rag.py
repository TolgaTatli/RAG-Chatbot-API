from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict

class HuggingFaceRAGQA:
    """Hugging Face Transformers ile ücretsiz RAG sistemi"""

    def __init__(self, retriever, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Args:
            retriever: RAG retriever instance
            model_name: HF model adı
        """
        self.retriever = retriever
        self.model_name = model_name
        self.generator = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Küçük modeller (RAM dostu)
        self.small_models = {
            "turkish": "microsoft/DialoGPT-small",  # Küçük model
            "multilingual": "microsoft/DialoGPT-medium",  # Orta model
            "code": "Salesforce/codegen-350M-mono",  # Kod için
            "chat": "microsoft/DialoGPT-large"  # Büyük model (daha iyi ama yavaş)
        }

    def load_model(self):
        """Modeli yükle"""
        try:
            print(f"🤖 Model yükleniyor: {self.model_name}")
            print("⏳ Bu işlem biraz zaman alabilir...")

            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )

            print(f"✅ Model başarıyla yüklendi ({self.device})")
            return True

        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            print("💡 İnternetsiz alternatif modeller deneyin")
            return False

    def generate_answer(self, question: str, context: str) -> str:
        """
        HuggingFace model ile cevap üret
        """
        if not self.generator:
            if not self.load_model():
                return "Model yüklenemedi. Lütfen model adını kontrol edin."

        prompt = f"""Bağlam: {context[:800]}

Soru: {question}

Cevap:"""

        try:
            response = self.generator(
                prompt,
                max_length=len(prompt.split()) + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )

            # Sadece yeni üretilen kısmı al
            generated_text = response[0]['generated_text']
            answer = generated_text[len(prompt):].strip()

            if not answer:
                return "Üzgünüm, bu soruya cevap oluşturamadım."

            return answer

        except Exception as e:
            return f"Cevap üretme hatası: {e}"

    def answer_question(self, question: str, top_k: int = 3) -> Dict:
        """Soruya cevap ver"""
        search_results = self.retriever.search(question, top_k)
        context = self.retriever.get_context_for_query(question, top_k)

        if not search_results:
            return {
                'question': question,
                'answer': "Bu soruyla ilgili bilgi bulamadım.",
                'context': "",
                'sources': [],
                'method': 'no_results'
            }

        if self.generator or self.load_model():
            answer = self.generate_answer(question, context)
            method = 'huggingface_generated'
        else:
            best_match = search_results[0]
            answer = f"En ilgili bilgi: {best_match['text'][:300]}..."
            method = 'retrieval_only'

        return {
            'question': question,
            'answer': answer,
            'context': context,
            'sources': search_results,
            'confidence': search_results[0]['score'] if search_results else 0,
            'method': method
        }

    def interactive_qa(self):
        """Etkileşimli soru-cevap"""
        print("🤗 Hugging Face RAG Sistemi")
        print(f"Model: {self.model_name}")
        print(f"Cihaz: {self.device}")
        print("-" * 50)

        while True:
            question = input("\n❓ Sorunuz: ").strip()

            if question.lower() in ['quit', 'çık', 'exit', 'q']:
                print("👋 Görüşmek üzere!")
                break

            if not question:
                continue

            try:
                print("🔍 Aranıyor ve cevap üretiliyor...")
                result = self.answer_question(question)

                print(f"\n✅ **Cevap:**")
                print(result['answer'])

                print(f"\n📊 **Bilgiler:**")
                print(f"• Yöntem: {result['method']}")
                print(f"• Güven: {result.get('confidence', 0):.3f}")

            except Exception as e:
                print(f"❌ Hata: {e}")

if __name__ == "__main__":
    from rag_system import RAGRetriever

    # Retriever'ı başlat
    retriever = RAGRetriever()

    try:
        retriever.load_from_files("faiss_index.bin", "documents.pkl")

        # HuggingFace QA sistemi oluştur
        qa_system = HuggingFaceRAGQA(retriever)
        qa_system.interactive_qa()

    except FileNotFoundError:
        print("❌ Index dosyaları bulunamadı.")
    except Exception as e:
        print(f"❌ Hata: {e}")
