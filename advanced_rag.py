import os
from typing import List, Dict
from openai import OpenAI
from rag_system import RAGRetriever
from dotenv import load_dotenv

# .env dosyasından environment variables yükle
load_dotenv()

class AdvancedRAGQA:
    """OpenAI GPT ile gelişmiş RAG soru-cevap sistemi"""

    def __init__(self, retriever: RAGRetriever, model_name: str = "gpt-3.5-turbo"):
        """
        Args:
            retriever: RAG retriever instance
            model_name: OpenAI model adı
        """
        self.retriever = retriever
        self.model_name = model_name

        # OpenAI istemcisini başlat
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
            print("UYARI: OPENAI_API_KEY environment variable bulunamadı!")
            print("Lütfen .env dosyasına OPENAI_API_KEY=your_api_key ekleyin")

    def generate_answer(self, question: str, context: str) -> str:
        """
        Bağlama dayalı cevap üret

        Args:
            question: Kullanıcı sorusu
            context: İlgili dökümanlardan oluşan bağlam

        Returns:
            Oluşturulan cevap
        """
        prompt = f"""Sen yardımсı bir asistansın. Aşağıdaki bağlam bilgilerini kullanarak kullanıcının sorusunu cevapla.

BAĞLAM:
{context}

SORU: {question}

KURALLAR:
1. Sadece verilen bağlam bilgilerini kullan
2. Bağlamda olmayan bilgileri uydurma
3. Eğer bağlamda cevap yoksa, bunu açıkça belirt
4. Türkçe cevap ver
5. Cevabını doğrudan ve anlaşılır şekilde ver

CEVAP:"""

        try:
            if not self.client:
                return "OpenAI API anahtarı bulunamadı. Lütfen .env dosyasını kontrol edin."

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Sen GEODI için özelleştirilmiş bir yapay zeka asistanısın. Verilen bağlam bilgilerini kullanarak soruları cevaplarsın."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.4
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Cevap oluştururken hata oluştu: {e}"

    def answer_question(self, question: str, top_k: int = 3) -> Dict:
        """
        Soruya detaylı cevap ver

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

        if not search_results:
            return {
                'question': question,
                'answer': "Üzgünüm, bu soruyla ilgili bilgi bulamadım.",
                'context': "",
                'sources': [],
                'method': 'no_results'
            }

        # GPT ile cevap oluştur
        if self.client:
            answer = self.generate_answer(question, context)
            method = 'gpt_generated'
        else:
            # Fallback: En iyi eşleşen dökümanı kullan
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
        """Gelişmiş etkileşimli soru-cevap modu"""
        print("🤖 Gelişmiş RAG Soru-Cevap Sistemi")
        print("Çıkmak için 'quit' yazın.")
        print("Yardım için 'help' yazın.")
        print("-" * 50)

        while True:
            question = input("\n❓ Sorunuz: ").strip()

            if question.lower() in ['quit', 'çık', 'exit', 'q']:
                print("👋 Görüşmek üzere!")
                break

            if question.lower() in ['help', 'yardım', 'h']:
                self.show_help()
                continue

            if not question:
                print("Lütfen bir soru yazın.")
                continue

            try:
                print("🔍 Aranıyor...")
                result = self.answer_question(question)

                print(f"\n✅ **Cevap:**")
                print(result['answer'])

                print(f"\n📊 **Bilgiler:**")
                print(f"• Güven skoru: {result.get('confidence', 0):.3f}")
                print(f"• Kaynak sayısı: {len(result['sources'])}")
                print(f"• Yöntem: {result['method']}")

                # Kaynakları göster seçeneği
                if result['sources']:
                    show_sources = input("\n📚 Kaynak dökümanları görmek ister misiniz? (e/h): ").strip().lower()
                    if show_sources in ['e', 'evet', 'y', 'yes']:
                        print("\n📖 **Kaynak Dökümanlar:**")
                        for i, source in enumerate(result['sources'], 1):
                            print(f"\n{i}. Kaynak (Skor: {source['score']:.3f})")
                            print(f"   {source['text'][:200]}...")

            except Exception as e:
                print(f"❌ Hata oluştu: {e}")

    def show_help(self):
        """Yardım bilgilerini göster"""
        help_text = """
🆘 **Yardım**

**Komutlar:**
• Normal soru sorun: Herhangi bir konu hakkında soru sorabilirsiniz
• 'help' veya 'yardım': Bu yardım metnini gösterir  
• 'quit' veya 'çık': Programdan çıkar

**İpuçları:**
• Spesifik sorular sorun, daha iyi sonuçlar alırsınız
• Uzun sorular yerine kısa ve net sorular tercih edin
• Türkçe veya İngilizce sorabilirsiniz

**Örnek sorular:**
• "Machine learning nedir?"
• "Python ile nasıl web scraping yapılır?"
• "Veri bilimi alanında hangi araçlar kullanılır?"
        """
        print(help_text)

if __name__ == "__main__":
    print("🚀 Gelişmiş RAG sistemi başlatılıyor...")

    # Retriever'ı başlat
    retriever = RAGRetriever()

    try:
        # Önceden oluşturulmuş index'i yükle
        retriever.load_from_files("faiss_index.bin", "documents.pkl")

        # Gelişmiş QA sistemi oluştur
        qa_system = AdvancedRAGQA(retriever)

        # Etkileşimli mod başlat
        qa_system.interactive_qa()

    except FileNotFoundError:
        print("❌ Index dosyaları bulunamadı. Önce data_processor.py çalıştırın.")
    except Exception as e:
        print(f"❌ Hata: {e}")
