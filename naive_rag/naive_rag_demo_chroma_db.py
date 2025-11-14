"""
Naive RAG - Базова реалізація
=============================
Проста імплементація RAG з використанням TF-IDF та косинусної подібності.
З використанням ChromaDB для збереження чанків та метаданих.

Точність: ~30% на складних запитах
"""
import sys
import os
from pathlib import Path
import time
import numpy as np
import hashlib
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
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
    except Exception as e:
        print(f"Помилка OpenAI: {e}")
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


def compute_file_checksum(file_path: str) -> str:
    """
    Обчислює SHA256 чексуму файлу для виявлення змін.
    
    Args:
        file_path: Шлях до файлу
        
    Returns:
        SHA256 чексума у вигляді hex рядка
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


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
    
    Використовує ChromaDB для persistent зберігання чанків та метаданих.
    """

    def __init__(
        self, 
        documents_path: str = "data/pdfs", 
        chunk_size: int = 500, 
        chunk_overlap: int = 100,
        chromadb_path: str = "naive_rag_chromadb"
    ):
        self.documents_path = documents_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.chunk_embeddings = []
        self.embeddings_model = SimpleEmbeddings()
        
        # ChromaDB для зберігання чанків
        self.chroma_client = None
        self.collection = None
       
        self._init_chromadb(Path(__file__).parent.parent / chromadb_path)

    def _init_chromadb(self, chromadb_path: str):
        try:
            # Створюємо persistent клієнт
            self.chroma_client = chromadb.PersistentClient(
                path=str(chromadb_path)
            )

            # Створюємо або завантажуємо collection
            try:
                self.collection = self.chroma_client.get_collection(name="naive_rag_chunks")
                print(f"✅ Завантажено існуючу ChromaDB collection")
            except:
                self.collection = self.chroma_client.create_collection(
                    name="naive_rag_chunks",
                    metadata={"description": "Naive RAG document chunks with TF-IDF"}
                )
                print("✅ Створено нову ChromaDB collection")
        except Exception as e:
            print(f"⚠️  Помилка ініціалізації ChromaDB: {e}")
            exit(1)

    def _is_document_chunks_exists(self, source: str, file_checksum: str, chunk_size: int, chunk_overlap: int) -> bool:
        """
        Перевіряє чи є вже чанки для документа в ChromaDB.
        
        Args:
            source: Ім'я файлу (source)
            
        Returns:
            True якщо є чанки, False якщо немає
        """
        if not self.collection:
            return False

        try:
            # Шукаємо всі чанки для цього документа
            results = self.collection.get(
                where={"$and": [{"source": source}, {"chunk_size": str(chunk_size)}, {"chunk_overlap": str(chunk_overlap)}, {"file_checksum": file_checksum}]},
            )

            if results['ids']:
                return True
        except Exception as e:
            print(f"⚠️  Помилка перевірки існуючих чанків для {source}: {e}")
        
        return False

    def _delete_chunks_by_source(self, source: str, file_checksum: str):
        """Видаляє всі чанки для документа з ChromaDB"""

        try:
            # Отримуємо всі ID чанків для цього документа
            results = self.collection.get(
                where={"source": source}
            )

            if results['ids']:
                for i, id in enumerate(results['ids']):
                    if results['metadatas'][i]['file_checksum'] != file_checksum:
                        self.chroma_client.delete_collection(id)
                        print(f"  🗑️  Видалено чанк {id} для {source} через зміну чексуми оригінального файлу")
        except Exception as e:
            print(f"⚠️  Помилка видалення чанків для {source}: {e}")

    def load_and_process_documents(self, max_documents=None):
        """
        Завантажує PDF файли та розбиває на чанки.
        Перевіряє чи вже є чанки в ChromaDB з такими ж параметрами та чексумою.
        Не завантажує всі чанки з БД в пам'ять - тільки зберігає інформацію про джерела.
        """
        loader = DocumentLoader(self.documents_path)
        documents = loader.load_documents(max_documents=max_documents)

        # Перевіряємо кожен документ на наявність в ChromaDB
        documents_to_process = []

        for doc in documents:
            source = doc["source"]
            file_path = doc.get("path", "")
            
            # Обчислюємо чексуму файлу
            if file_path and Path(file_path).exists():
                current_checksum = compute_file_checksum(file_path)
            else:
                current_checksum = ""

            # Перевіряємо чи є вже чанки в БД
            self._delete_chunks_by_source(source, current_checksum)
            _is_document_chunks_exists = self._is_document_chunks_exists(source, current_checksum, self.chunk_size, self.chunk_overlap)

            if _is_document_chunks_exists:
                print(f"  ✓ {source}: чанки вже є в БД")
                continue
            
            documents_to_process.append(doc)

        if documents_to_process:
            print(f"  📝 Потрібно обробити {len(documents_to_process)} документів")

        # Розбиваємо на чанки тільки нові/змінені документи
        new_chunks = []
        if documents_to_process:
            splitter = TextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            new_chunks = splitter.split_documents(documents_to_process)
            
            # Додаємо чексуму до метаданих чанків
            for chunk in new_chunks:
                source = chunk["source"]
                # Знаходимо відповідний документ
                doc = next((d for d in documents_to_process if d["source"] == source), None)
                if doc:
                    file_path = doc.get("path", "")
                    if file_path and Path(file_path).exists():
                        chunk["file_checksum"] = compute_file_checksum(file_path)
                    else:
                        chunk["file_checksum"] = ""

        # Зберігаємо тільки нові чанки (не завантажуємо всі з БД)
        self.chunks = new_chunks

        if new_chunks:
            print(f"  📝 Потрібно зберегти {len(new_chunks)} чанків")
            self._save_new_chunks_to_db(new_chunks)

        return documents

    def _find_chunks_by_source_any_checksum(self, source: str) -> bool:
        """
        Перевіряє чи є чанки для джерела в БД (незалежно від чексуми).
        
        Returns:
            True якщо є чанки, False якщо немає
        """
        if not self.collection:
            return False

        try:
            results = self.collection.get(
                where={"source": source},
                limit=1  # Тільки перевіряємо наявність
            )
            return len(results.get('ids', [])) > 0
        except Exception as e:
            return False

    def _load_chunks_from_db(self, source: str, limit: int = None) -> List[Dict]:
        """
        Завантажує чанки для документа з ChromaDB (lazy loading).
        
        Args:
            source: Ім'я файлу
            limit: Максимальна кількість чанків (None = всі)
        """
        if not self.collection:
            return []

        try:
            # Отримуємо чанки для цього документа
            query_limit = limit if limit else 10000
            results = self.collection.get(
                where={"source": source},
                limit=query_limit
            )
            
            chunks = []
            if results['ids']:
                for i, chunk_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]
                    chunk = {
                        "content": results['documents'][i],
                        "source": metadata.get("source", source),
                        "chunk_id": int(metadata.get("chunk_id", 0)),
                        "total_chunks": int(metadata.get("total_chunks", 0)),
                        "file_checksum": metadata.get("file_checksum", ""),
                        "file_path": metadata.get("file_path", "")
                    }
                    chunks.append(chunk)
                
                # Сортуємо за chunk_id
                chunks.sort(key=lambda x: x["chunk_id"])
            
            return chunks
        except Exception as e:
            print(f"⚠️  Помилка завантаження чанків з БД для {source}: {e}")
            return []

    def _save_new_chunks_to_db(self, chunks: List[Dict]):
        """
        Зберігає нові чанки в ChromaDB.
        """

        # Визначаємо які чанки потрібно зберегти (тільки нові, що мають file_checksum)
        chunks_to_save = []
        for chunk in chunks:
            # Використовуємо унікальний ID: source + chunk_id
            chunk_id_str = f"{chunk['source']}_chunk_{chunk.get('chunk_id', 0)}"
            chunks_to_save.append((chunk_id_str, chunk))

        if not chunks_to_save:
            return

        # Готуємо дані для збереження
        ids = []
        documents = []
        metadatas = []

        for chunk_id_str, chunk in chunks_to_save:
            ids.append(chunk_id_str)
            documents.append(chunk["content"])
            metadatas.append({
                "source": chunk["source"],
                "chunk_id": str(chunk.get("chunk_id", 0)),
                "chunk_size": str(self.chunk_size),
                "chunk_overlap": str(self.chunk_overlap),
                "file_checksum": chunk.get("file_checksum", ""),
            })

        # Зберігаємо батчами
        batch_size = 100
        saved_count = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]

            try:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_meta
                )
                saved_count += len(batch_ids)
            except Exception as e:
                # Можливо чанк вже існує - це нормально
                if "duplicate" not in str(e).lower():
                    print(f"⚠️  Помилка збереження чанків в БД: {e}")

        if saved_count > 0:
            print(f"  💾 Збережено {saved_count} нових чанків в ChromaDB")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Знаходить найбільш релевантні чанки через косинусну подібність.
        Завантажує чанки з БД тільки при потребі (lazy loading).

        Args:
            query: Запитання користувача
            top_k: Кількість чанків для повернення

        Returns:
            Список топ-k найбільш схожих чанків з оцінками
        """
        top_chunks = self.chroma_client.get_collection(name="naive_rag_chunks").query(
            query_texts=[query],
            n_results=top_k,
            where={"$and": [{"chunk_size": str(self.chunk_size)}, {"chunk_overlap": str(self.chunk_overlap)}]},
        )

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

        # ChromaDB returns nested lists: documents[0] is list of documents for first query
        documents = context_chunks.get("documents", [[]])
        contexts = documents[0] if documents and documents[0] else []

        # Використовуємо LLM для генерації
        answer = generate_answer_with_llm(
            question=query,
            contexts=contexts,
            max_tokens=256,
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

        # ChromaDB returns nested lists: distances[0] is list of distances for first query
        distances = relevant_chunks.get("distances", [[]])
        ids = relevant_chunks.get("ids", [[]])
        documents = relevant_chunks.get("documents", [[]])
        
        result = {
            "question": question,
            "answer": answer,
            "relevant_chunks": len(distances[0]) if distances and distances[0] else 0,
            "sources": ids[0] if ids and ids[0] else [],
            "scores": distances[0] if distances and distances[0] else [],
            "contexts": documents[0] if documents and documents[0] else [],
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
    print(f"Завантажено: {len(documents)} документів")

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
            score = result['scores'][0] if result['scores'] else 0.0
            print(f"  Час: {result['execution_time']:.2f}с | Оцінка: {score:.3f}")

    # Розраховуємо підсумкову статистику
    avg_time = np.mean([q["execution_time"] for q in all_results["queries"]])
    avg_score = np.mean([q["scores"][0] if q["scores"] else 0.0 for q in all_results["queries"]])

    all_results["metrics"] = {
        "average_execution_time": avg_time,
        "average_top_score": avg_score,
        "total_queries": len(all_results["queries"])
    }

    # Зберігаємо результати
    save_results(all_results, "results/naive_rag_chroma_db_results.json")

    # Виводимо підсумок
    print("\n" + "="*70)
    print("ПІДСУМОК")
    print("="*70)
    print(f"Всього запитів: {len(all_results['queries'])}")
    print(f"Середній час виконання: {avg_time:.2f}с")
    print(f"Середня оцінка: {avg_score:.3f}")
    print(f"\nРезультати збережено: results/naive_rag_chroma_db_results.json")

    print("\n" + "="*70)
    print("Обмеження Naive RAG:")
    print("  - Низька точність на складних запитах (~30%)")
    print("  - Відсутність контексту між чанками")
    print("  - Немає перевірки релевантності")
    print("  - Проблема 'Lost in the Middle'")
    print("="*70)


if __name__ == "__main__":
    run_naive_rag_demo()
