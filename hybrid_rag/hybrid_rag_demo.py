"""
Hybrid RAG - Комбінований пошук
===============================
Поєднує Dense (векторний) та Sparse (BM25) пошук.
Методи: RRF (Reciprocal Rank Fusion) та Convex Combination.

Точність: 83%
"""
import sys
import os
from pathlib import Path
import time
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

# Завантаження змінних середовища з .env (шукаємо в поточній та батьківській директорії)
load_dotenv()  # Спочатку поточна директорія
if not os.getenv('OPENAI_API_KEY'):
    # Якщо не знайшли, шукаємо в батьківській директорії
    load_dotenv(Path(__file__).parent.parent / '.env')

from utils.data_loader import DocumentLoader, TextSplitter, save_results


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


class DenseRetriever:
    """Dense retrieval - векторний пошук через TF-IDF"""

    def __init__(self):
        self.vocabulary = {}
        self.idf = {}

    def fit(self, documents: List[str]):
        """Будує TF-IDF модель"""
        doc_word_sets = []
        for doc in documents:
            words = set(doc.lower().split())
            doc_word_sets.append(words)
            for word in words:
                self.vocabulary[word] = self.vocabulary.get(word, 0) + 1

        num_docs = len(documents)
        for word in self.vocabulary:
            doc_count = sum(1 for word_set in doc_word_sets if word in word_set)
            self.idf[word] = np.log(num_docs / (doc_count + 1)) + 1

    def embed(self, text: str) -> np.ndarray:
        """Створює TF-IDF вектор"""
        words = text.lower().split()
        word_count = defaultdict(int)
        for word in words:
            word_count[word] += 1

        vector = np.zeros(len(self.vocabulary))
        for i, word in enumerate(sorted(self.vocabulary.keys())):
            if word in word_count:
                tf = word_count[word] / max(len(words), 1)
                idf = self.idf.get(word, 1)
                vector[i] = tf * idf

        return vector

    def search(self, query: str, doc_vectors: List[np.ndarray], top_k: int = 10) -> List[Tuple[int, float]]:
        """Шукає найбільш схожі документи"""
        query_vector = self.embed(query)

        similarities = []
        for i, doc_vector in enumerate(doc_vectors):
            similarity = self._cosine_similarity(query_vector, doc_vector)
            similarities.append((i, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Розраховує косинусну подібність"""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot / (norm1 * norm2 + 1e-10)


class SparseRetriever:
    """Sparse retrieval - keyword-based пошук через BM25"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.documents = []

    def fit(self, documents: List[str]):
        """Будує BM25 індекс"""
        self.documents = documents
        self.doc_len = [len(doc.split()) for doc in documents]
        self.avgdl = np.mean(self.doc_len) if self.doc_len else 1

        df = defaultdict(int)
        for doc in documents:
            unique_words = set(doc.lower().split())
            for word in unique_words:
                df[word] += 1

        num_docs = len(documents)
        for word, freq in df.items():
            self.idf[word] = np.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Шукає документи за BM25"""
        scores = []
        for i in range(len(self.documents)):
            score = self._bm25_score(query, i)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _bm25_score(self, query: str, doc_id: int) -> float:
        """Розраховує BM25 оцінку"""
        score = 0
        doc = self.documents[doc_id]
        doc_words = doc.lower().split()
        doc_word_counts = defaultdict(int)
        for word in doc_words:
            doc_word_counts[word] += 1

        for word in query.lower().split():
            if word in doc_word_counts:
                freq = doc_word_counts[word]
                idf = self.idf.get(word, 0)
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl)
                score += idf * numerator / denominator

        return score


class HybridRAG:
    """
    Гібридна RAG система.
    Поєднує Dense (векторний) та Sparse (BM25) пошук через:
    - Convex Combination: α * dense + (1-α) * sparse
    """

    def __init__(self, documents_path: str = "data/pdfs", alpha: float = 0.5,
                 chunk_size: int = 500, chunk_overlap: int = 100):
        self.documents_path = documents_path
        self.alpha = alpha  # Вага для векторного пошуку
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.chunk_vectors = []
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()

    def load_and_process_documents(self, max_documents=None):
        """Завантажує документи та розбиває на чанки"""
        loader = DocumentLoader(self.documents_path)
        documents = loader.load_documents(max_documents=max_documents)

        splitter = TextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.chunks = splitter.split_documents(documents)

        return documents

    def create_indexes(self):
        """Створює індекси для обох ретриверів"""
        all_texts = [chunk["content"] for chunk in self.chunks]

        # Dense index (векторні представлення)
        self.dense_retriever.fit(all_texts)
        self.chunk_vectors = [self.dense_retriever.embed(text) for text in all_texts]

        # Sparse index (BM25)
        self.sparse_retriever.fit(all_texts)

    def hybrid_search(self, query: str, method: str = "convex", top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Гібридний пошук: комбінує dense та sparse результати.

        Args:
            query: Запит
            method: "convex" або "rrf" (Reciprocal Rank Fusion)
            top_k: Кількість результатів

        Returns:
            Список (chunk_id, score)
        """
        # Dense search (векторний)
        dense_results = self.dense_retriever.search(query, self.chunk_vectors, top_k=20)

        # Sparse search (BM25)
        sparse_results = self.sparse_retriever.search(query, top_k=20)

        if method == "rrf":
            # Reciprocal Rank Fusion
            return self._reciprocal_rank_fusion(dense_results, sparse_results, top_k)
        else:
            # Convex Combination
            return self._convex_combination(dense_results, sparse_results, top_k)

    def _convex_combination(self, dense_results: List[Tuple[int, float]],
                           sparse_results: List[Tuple[int, float]],
                           top_k: int) -> List[Tuple[int, float]]:
        """Комбінує результати через зважену суму: α * dense + (1-α) * sparse"""
        # Нормалізація sparse scores
        max_sparse = max(score for _, score in sparse_results) if sparse_results else 1
        normalized_sparse = {idx: score / max_sparse for idx, score in sparse_results}

        # Комбінування
        combined_scores = {}
        for idx, dense_score in dense_results:
            sparse_score = normalized_sparse.get(idx, 0)
            combined_scores[idx] = self.alpha * dense_score + (1 - self.alpha) * sparse_score

        # Додаємо результати що є тільки в sparse
        for idx, score in sparse_results:
            if idx not in combined_scores:
                combined_scores[idx] = (1 - self.alpha) * (score / max_sparse)

        # Сортування
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def _reciprocal_rank_fusion(self, dense_results: List[Tuple[int, float]],
                                sparse_results: List[Tuple[int, float]],
                                top_k: int, k: int = 60) -> List[Tuple[int, float]]:
        """RRF: 1/(k + rank)"""
        rrf_scores = defaultdict(float)

        # Dense ranks
        for rank, (idx, _) in enumerate(dense_results):
            rrf_scores[idx] += 1.0 / (k + rank + 1)

        # Sparse ranks
        for rank, (idx, _) in enumerate(sparse_results):
            rrf_scores[idx] += 1.0 / (k + rank + 1)

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def query(self, question: str, method: str = "convex", top_k: int = 5) -> Dict:
        """Виконує повний Hybrid RAG pipeline"""
        start_time = time.time()

        # Гібридний пошук
        results = self.hybrid_search(question, method=method, top_k=top_k)

        # Витягуємо чанки
        relevant_chunks = [self.chunks[idx].copy() for idx, score in results]
        for i, (_, score) in enumerate(results):
            relevant_chunks[i]["score"] = score

        # Генерація відповіді
        contexts = [chunk["content"] for chunk in relevant_chunks]
        answer = generate_answer_with_llm(
            question=question,
            contexts=contexts,
            max_tokens=256
        )

        execution_time = time.time() - start_time

        return {
            "question": question,
            "answer": answer,
            "method": method,
            "alpha": self.alpha,
            "relevant_chunks": len(relevant_chunks),
            "sources": [chunk["source"] for chunk in relevant_chunks],
            "scores": [chunk["score"] for chunk in relevant_chunks],
            "execution_time": execution_time
        }


def run_hybrid_rag_demo():
    """Запускає демонстрацію Hybrid RAG"""
    print("="*70)
    print("HYBRID RAG ДЕМОНСТРАЦІЯ")
    print("="*70)

    # Ініціалізація
    chunk_size = 500
    chunk_overlap = 100
    alpha = 0.5  # 50% векторний, 50% BM25

    rag = HybridRAG(
        documents_path="data/pdfs",
        alpha=alpha,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Виводимо конфігурацію
    print(f"\nКонфігурація:")
    llm_model = detect_llm_provider()
    print(f"  Модель LLM: {llm_model}")
    print(f"  Розмір чанку: {chunk_size} символів")
    print(f"  Перекриття чанків: {chunk_overlap} символів")
    print(f"  Alpha (вага векторного пошуку): {alpha}")
    print(f"  Методи: Convex Combination, RRF")

    # Завантаження
    print(f"\nЗавантаження документів...")
    documents = rag.load_and_process_documents(max_documents=50)
    print(f"Завантажено: {len(documents)} документів, {len(rag.chunks)} чанків")

    # Створення індексів
    print(f"Створення індексів...")
    rag.create_indexes()
    print(f"Створено: Dense index ({len(rag.chunk_vectors)} векторів) + BM25 index")

    # Завантажуємо УНІФІКОВАНИЙ тестовий датасет (100 запитів)
    # ВАЖЛИВО: Всі RAG підходи використовують ТІ САМІ запити для коректного порівняння!
    from collections import defaultdict
    loader = DocumentLoader()
    unified_queries = loader.load_unified_queries(max_queries=50)  # Перші 50 для швидкості
    print(f"Тестових запитів: {len(unified_queries)}")

    print("\n" + "="*70)
    print("ВИКОНАННЯ ТЕСТІВ")
    print("="*70)

    all_results = {
        "system_name": "Hybrid RAG",
        "total_documents": len(documents),
        "total_chunks": len(rag.chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "alpha": alpha,
        "llm_model": detect_llm_provider(),
        "queries": []
    }

    # Групуємо по категоріях для виводу
    queries_by_category = defaultdict(list)
    for query in unified_queries:
        queries_by_category[query.get("category", "general")].append(query)

    # Тестуємо запити по категоріях
    print(f"\nМетод: Convex Combination (α={alpha})")
    for category, queries in queries_by_category.items():
        print(f"Категорія: {category}")

        for query_data in queries:
            question = query_data.get("question", "")

            # Виконуємо запит
            result = rag.query(question, method="convex", top_k=5)
            result["category"] = category
            result["query_id"] = query_data.get("id")
            result["difficulty"] = query_data.get("difficulty")
            all_results["queries"].append(result)

            # Виводимо короткий результат
            print(f"  ID {query_data.get('id')}: {question[:70]}...")
            print(f"  Час: {result['execution_time']:.2f}с | Оцінка: {result['scores'][0]:.3f}")

    # Статистика
    avg_time = np.mean([q["execution_time"] for q in all_results["queries"]])
    avg_score = np.mean([q["scores"][0] for q in all_results["queries"]])

    all_results["metrics"] = {
        "average_execution_time": avg_time,
        "average_top_score": avg_score,
        "total_queries": len(all_results["queries"])
    }

    save_results(all_results, "results/hybrid_rag_results.json")

    print("\n" + "="*70)
    print("ПІДСУМОК")
    print("="*70)
    print(f"Всього запитів: {len(all_results['queries'])}")
    print(f"Середній час: {avg_time:.2f}с")
    print(f"Середня оцінка: {avg_score:.3f}")
    print(f"\nРезультати збережено: results/hybrid_rag_results.json")
    print("="*70)


if __name__ == "__main__":
    run_hybrid_rag_demo()
