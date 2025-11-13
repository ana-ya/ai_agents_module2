"""
Naive RAG - Базова реалізація
=============================
Проста імплементація RAG з використанням TF-IDF та косинусної подібності.

Точність: ~30% на складних запитах
"""
import sys
import os
from pathlib import Path
import time
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv

# Додаємо шлях до утиліт
sys.path.append(str(Path(__file__).parent.parent))

# Завантаження змінних середовища з .env (шукаємо в поточній та батьківській директорії)
load_dotenv()  # Спочатку поточна директорія
if not os.getenv('OPENAI_API_KEY'):
    # Якщо не знайшли, шукаємо в батьківській директорії
    load_dotenv(Path(__file__).parent.parent / '.env')

from utils.data_loader import DocumentLoader, TextSplitter, save_results, print_results


def generate_answer_with_llm(question: str, contexts: List[str], max_tokens: int = 256) -> str:
    """
    Генерація відповіді через LLM
    Спроба 1: Ollama (локально, безкоштовно)
    Спроба 2: OpenAI (якщо є API key), зробіть export OPENAI_API_KEY=your_key
    Спроба 3: Simple fallback - повернути контекст
    """
    # Спроба 1: Ollama (локально)
    try:
        import requests
        prompt = f"Based on the following context, answer the question.\n\nContext:\n{chr(10).join(contexts[:3])}\n\nQuestion: {question}\n\nAnswer:"

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": max_tokens}
            },
            timeout=30
        )

        if response.status_code == 200:
            return response.json()["response"].strip()
    except Exception:
        pass

    # Спроба 2: OpenAI (якщо є API key)
    # Для використання: export OPENAI_API_KEY=your_key
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            prompt = f"Based on the following context, answer the question.\n\nContext:\n{chr(10).join(contexts[:3])}\n\nQuestion: {question}"

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
    except Exception:
        pass

    # Спроба 3: Fallback - просто повернути контекст
    return "\n\n".join(contexts[:3]) if contexts else "Не знайдено релевантної інформації."


def detect_llm_provider() -> str:
    """Визначає який LLM provider доступний"""
    # Перевіряємо Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            return "ollama (llama3.2:3b)"
    except:
        pass

    # Перевіряємо OpenAI
    if os.getenv("OPENAI_API_KEY"):
        return "openai (gpt-4o-mini)"

    return "fallback (без LLM)"


class SimpleEmbeddings:
    """
    Проста реалізація TF-IDF векторизації.
    Не потребує зовнішніх API - працює на numpy.
    """

    def __init__(self):
        self.vocabulary = {}
        self.idf = {}

    def fit(self, documents: List[str]):
        """Будує словник та розраховує IDF значення"""
        # Будуємо словник зі всіх документів
        doc_word_sets = []
        for doc in documents:
            words = set(doc.lower().split())
            doc_word_sets.append(words)
            for word in words:
                self.vocabulary[word] = self.vocabulary.get(word, 0) + 1

        # Розраховуємо IDF (Inverse Document Frequency) для кожного слова
        num_docs = len(documents)
        for word in self.vocabulary:
            doc_count = sum(1 for word_set in doc_word_sets if word in word_set)
            self.idf[word] = np.log(num_docs / (doc_count + 1))

    def embed(self, text: str) -> np.ndarray:
        """Створює TF-IDF вектор для тексту"""
        words = text.lower().split()
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1

        # Будуємо TF-IDF вектор
        vector = np.zeros(len(self.vocabulary))
        for i, word in enumerate(sorted(self.vocabulary.keys())):
            if word in word_count:
                tf = word_count[word] / len(words)
                idf = self.idf.get(word, 0)
                vector[i] = tf * idf

        return vector

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Розраховує косинусну подібність між двома векторами"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class NaiveRAG:
    """
    Базова RAG система з трьома основними компонентами:
    1. Розбиття документів на чанки
    2. TF-IDF векторний пошук
    3. Генерація відповіді через LLM
    """

    def __init__(self, documents_path: str = "data/pdfs", chunk_size: int = 500, chunk_overlap: int = 100):
        self.documents_path = documents_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.chunk_embeddings = []
        self.embeddings_model = SimpleEmbeddings()

    def load_and_process_documents(self, max_documents=None):
        """Завантажує PDF файли та розбиває на чанки"""
        loader = DocumentLoader(self.documents_path)
        documents = loader.load_documents(max_documents=max_documents)

        # Розбиваємо на чанки з перекриттям
        splitter = TextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.chunks = splitter.split_documents(documents)

        return documents

    def create_embeddings(self):
        """Генерує TF-IDF векторні представлення для всіх чанків"""
        # Тренуємо модель на всіх чанках
        all_texts = [chunk["content"] for chunk in self.chunks]
        self.embeddings_model.fit(all_texts)

        # Створюємо векторні представлення
        self.chunk_embeddings = []
        for chunk in self.chunks:
            embedding = self.embeddings_model.embed(chunk["content"])
            self.chunk_embeddings.append(embedding)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Знаходить найбільш релевантні чанки через косинусну подібність.

        Args:
            query: Запитання користувача
            top_k: Кількість чанків для повернення

        Returns:
            Список топ-k найбільш схожих чанків з оцінками
        """
        # Створюємо вектор для запиту
        query_embedding = self.embeddings_model.embed(query)

        # Розраховуємо схожість з усіма чанками
        similarities = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = self.embeddings_model.cosine_similarity(
                query_embedding,
                chunk_embedding
            )
            similarities.append((i, similarity))

        # Сортуємо за схожістю (спадання)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Повертаємо топ-k
        top_chunks = []
        for i, similarity in similarities[:top_k]:
            chunk = self.chunks[i].copy()
            chunk["similarity_score"] = similarity
            top_chunks.append(chunk)

        return top_chunks

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Генерує відповідь через LLM використовуючи знайдений контекст.

        Args:
            query: Запитання
            context_chunks: Знайдені релевантні чанки

        Returns:
            Згенерована відповідь
        """
        if not context_chunks:
            return "Не знайдено релевантної інформації."

        # Витягуємо текстовий контент
        contexts = [chunk["content"] for chunk in context_chunks]

        # Використовуємо LLM для генерації
        answer = generate_answer_with_llm(
            question=query,
            contexts=contexts,
            max_tokens=256
        )

        return answer

    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        Виконує повний RAG pipeline: пошук + генерація.

        Args:
            question: Запитання користувача
            top_k: Кількість чанків для пошуку

        Returns:
            Словник з відповіддю та метаданими
        """
        start_time = time.time()

        # Крок 1: Пошук релевантних чанків
        relevant_chunks = self.retrieve(question, top_k=top_k)

        # Крок 2: Генерація відповіді
        answer = self.generate_answer(question, relevant_chunks)

        execution_time = time.time() - start_time

        result = {
            "question": question,
            "answer": answer,
            "relevant_chunks": len(relevant_chunks),
            "sources": [chunk["source"] for chunk in relevant_chunks],
            "scores": [chunk["similarity_score"] for chunk in relevant_chunks],
            "contexts": [chunk["content"] for chunk in relevant_chunks],
            "execution_time": execution_time
        }

        return result


def run_naive_rag_demo():
    """Запускає демонстрацію Naive RAG з тестовими запитами"""
    print("="*70)
    print("NAIVE RAG ДЕМОНСТРАЦІЯ")
    print("="*70)

    # Ініціалізація системи
    chunk_size = 500
    chunk_overlap = 100
    rag = NaiveRAG(
        documents_path="data/pdfs",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Виводимо конфігурацію
    print(f"\nКонфігурація:")
    llm_model = detect_llm_provider()
    print(f"  Модель LLM: {llm_model}")
    print(f"  Розмір чанку: {chunk_size} символів")
    print(f"  Перекриття чанків: {chunk_overlap} символів")

    # Завантажуємо документи
    # Примітка: max_documents=50 для швидкого демо. None - всі 660 документів.
    print(f"\nЗавантаження документів...")
    documents = rag.load_and_process_documents(max_documents=50)
    print(f"Завантажено: {len(documents)} документів, {len(rag.chunks)} чанків")

    # Створюємо векторні представлення
    print(f"Створення embeddings...")
    rag.create_embeddings()
    print(f"Створено: {len(rag.chunk_embeddings)} векторів")

    # Завантажуємо УНІФІКОВАНИЙ тестовий датасет (100 запитів)
    # ВАЖЛИВО: Всі RAG підходи використовують ТІ САМІ запити для коректного порівняння!
    loader = DocumentLoader()
    unified_queries = loader.load_unified_queries(max_queries=50)  # Перші 50 для швидкості
    print(f"Тестових запитів: {len(unified_queries)}")

    # Запускаємо тести
    print("\n" + "="*70)
    print("ВИКОНАННЯ ТЕСТІВ")
    print("="*70)

    all_results = {
        "system_name": "Naive RAG",
        "total_documents": len(documents),
        "total_chunks": len(rag.chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "llm_model": detect_llm_provider(),
        "queries": []
    }

    # Групуємо по категоріях для виводу
    from collections import defaultdict
    queries_by_category = defaultdict(list)
    for query in unified_queries:
        queries_by_category[query.get("category", "general")].append(query)

    # Тестуємо запити по категоріях
    for category, queries in queries_by_category.items():
        print(f"\nКатегорія: {category}")

        for query_data in queries:
            question = query_data.get("question", "")

            # Виконуємо запит
            result = rag.query(question, top_k=3)
            result["category"] = category
            result["query_id"] = query_data.get("id")
            result["difficulty"] = query_data.get("difficulty")
            all_results["queries"].append(result)

            # Виводимо короткий результат
            print(f"  ID {query_data.get('id')}: {question[:70]}...")
            print(f"  Час: {result['execution_time']:.2f}с | Оцінка: {result['scores'][0]:.3f}")

    # Розраховуємо підсумкову статистику
    avg_time = np.mean([q["execution_time"] for q in all_results["queries"]])
    avg_score = np.mean([q["scores"][0] for q in all_results["queries"]])

    all_results["metrics"] = {
        "average_execution_time": avg_time,
        "average_top_score": avg_score,
        "total_queries": len(all_results["queries"])
    }

    # Зберігаємо результати
    save_results(all_results, "results/naive_rag_results.json")

    # Виводимо підсумок
    print("\n" + "="*70)
    print("ПІДСУМОК")
    print("="*70)
    print(f"Всього запитів: {len(all_results['queries'])}")
    print(f"Середній час виконання: {avg_time:.2f}с")
    print(f"Середня оцінка: {avg_score:.3f}")
    print(f"\nРезультати збережено: results/naive_rag_results.json")

    print("\n" + "="*70)
    print("Обмеження Naive RAG:")
    print("  - Низька точність на складних запитах (~30%)")
    print("  - Відсутність контексту між чанками")
    print("  - Немає перевірки релевантності")
    print("  - Проблема 'Lost in the Middle'")
    print("="*70)


if __name__ == "__main__":
    run_naive_rag_demo()
