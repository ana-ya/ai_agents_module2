"""
LLM Helper для RAG демонстрацій
Підтримка Ollama (локальний) та OpenAI (fallback)
"""
import requests
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Завантаження змінних середовища з .env (шукаємо в поточній та батьківській директорії)
load_dotenv()  # Спочатку поточна директорія
if not os.getenv('OPENAI_API_KEY'):
    # Якщо не знайшли, шукаємо в батьківській директорії
    load_dotenv(Path(__file__).parent.parent / '.env')


class LLMGenerator:
    """
    Універсальний LLM генератор з підтримкою Ollama та OpenAI
    """

    def __init__(self, prefer_ollama: bool = True):
        """
        Args:
            prefer_ollama: Спробувати Ollama спочатку (True), або тільки OpenAI (False)
        """
        self.prefer_ollama = prefer_ollama
        self.ollama_available = self._check_ollama()
        self.openai_available = self._check_openai()

        if self.prefer_ollama and self.ollama_available:
            self.provider = "ollama"
            print("🤖 LLM Provider: Ollama (llama3.2:3b)")
        elif self.openai_available:
            self.provider = "openai"
            print("🤖 LLM Provider: OpenAI (gpt-4o-mini)")
        else:
            self.provider = "none"
            print("⚠️  LLM не доступний! Використовується простий concatenation")

    def _check_ollama(self) -> bool:
        """Перевірити чи доступний Ollama"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                # Шукаємо llama3.2:3b або будь-яку llama модель
                for model in models:
                    if "llama" in model.get("name", "").lower():
                        return True
            return False
        except:
            return False

    def _check_openai(self) -> bool:
        """Перевірити чи є OpenAI API key"""
        return os.getenv("OPENAI_API_KEY") is not None

    def generate_answer(
        self,
        question: str,
        contexts: List[str],
        max_tokens: int = 256
    ) -> str:
        """
        Згенерувати відповідь на основі питання та контекстів

        Args:
            question: Запитання користувача
            contexts: Список знайдених документів/чанків
            max_tokens: Максимальна довжина відповіді

        Returns:
            Згенерована відповідь
        """
        if self.provider == "ollama":
            return self._generate_with_ollama(question, contexts, max_tokens)
        elif self.provider == "openai":
            return self._generate_with_openai(question, contexts, max_tokens)
        else:
            # Fallback: простий concatenation
            return self._simple_concatenation(contexts)

    def _generate_with_ollama(
        self,
        question: str,
        contexts: List[str],
        max_tokens: int
    ) -> str:
        """Генерація через Ollama"""
        # Побудова промпту
        context_text = "\n\n".join([f"[Document {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])

        prompt = f"""Based on the following documents, please answer the question.
If the answer is not in the documents, say "I don't have enough information to answer this question."

Documents:
{context_text}

Question: {question}

Answer:"""

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": max_tokens
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                print(f"⚠️  Ollama error: {response.status_code}")
                return self._simple_concatenation(contexts)

        except Exception as e:
            print(f"⚠️  Ollama exception: {e}")
            return self._simple_concatenation(contexts)

    def _generate_with_openai(
        self,
        question: str,
        contexts: List[str],
        max_tokens: int
    ) -> str:
        """Генерація через OpenAI"""
        try:
            from openai import OpenAI
            client = OpenAI()

            context_text = "\n\n".join([f"[Document {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided documents. If the answer is not in the documents, say so."
                    },
                    {
                        "role": "user",
                        "content": f"""Documents:
{context_text}

Question: {question}

Please provide a concise answer based only on the information in the documents."""
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"⚠️  OpenAI exception: {e}")
            return self._simple_concatenation(contexts)

    def _simple_concatenation(self, contexts: List[str]) -> str:
        """Fallback: простий concatenation без LLM"""
        if not contexts:
            return "No relevant information found."

        # Обмежуємо кожен context до 200 символів
        truncated = [ctx[:200] + "..." if len(ctx) > 200 else ctx for ctx in contexts[:3]]
        return "\n\n".join(truncated)


# Глобальний instance для переви використання
_llm_generator = None

def get_llm_generator(prefer_ollama: bool = True) -> LLMGenerator:
    """
    Отримати глобальний LLM generator (singleton pattern)

    Args:
        prefer_ollama: Спробувати Ollama спочатку

    Returns:
        LLMGenerator instance
    """
    global _llm_generator
    if _llm_generator is None:
        _llm_generator = LLMGenerator(prefer_ollama=prefer_ollama)
    return _llm_generator
