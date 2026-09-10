from __future__ import annotations

import os
import re
from typing import Iterable

import requests

from backend.app.config.settings import GROQ_API_KEY
from backend.app.services.settings_service import get_settings
from backend.app.ml.prompts.templates import (
    BASE_SYSTEM_PROMPT,
    build_chat_messages,
    build_lifestyle_prompt,
    build_report_explanation_prompt,
)


class LLMService:
    def __init__(self, provider: str | None = None) -> None:
        settings = get_settings()

        self.provider = (
            provider
            or settings.get("ai_provider")
            or "groq"
        ).lower()

        # Current lightweight Groq production model.
        # Can be overridden in Render with MEDILENS_GROQ_MODEL.
        self.groq_model = os.getenv(
            "MEDILENS_GROQ_MODEL",
            "openai/gpt-oss-20b",
        )

        self.openai_model = os.getenv(
            "MEDILENS_OPENAI_MODEL",
            "gpt-4o-mini",
        )

        self.ollama_model = os.getenv(
            "MEDILENS_OLLAMA_MODEL",
            "llama3.1",
        )

        self.temperature = float(
            os.getenv(
                "MEDILENS_LLM_TEMPERATURE",
                "0.2",
            )
        )

        # Prevent an external AI provider from hanging the
        # Render request indefinitely.
        self.request_timeout = float(
            os.getenv(
                "MEDILENS_LLM_TIMEOUT",
                "30",
            )
        )

    # ------------------------------------------------------------------
    # Report explanation
    # ------------------------------------------------------------------

    def explain_report(
        self,
        *,
        disease_name: str,
        probability: float,
        top_features: list[dict[str, object]],
        report_context: str,
        history: Iterable[dict[str, str]] | None = None,
    ) -> str:
        prompt = build_report_explanation_prompt(
            disease_name=disease_name,
            probability=probability,
            top_features=top_features,
            report_context=report_context,
        )

        messages = build_chat_messages(
            user_message=prompt,
            report_context=report_context,
            knowledge_context=self._feature_context(
                top_features
            ),
            history=history,
        )

        return self.generate(
            messages,
            fallback_context=prompt,
        )

    # ------------------------------------------------------------------
    # Lifestyle recommendations
    # ------------------------------------------------------------------

    def lifestyle_recommendations(
        self,
        *,
        disease_name: str,
        report_context: str,
        top_features: list[dict[str, object]],
    ) -> list[str]:
        prompt = build_lifestyle_prompt(
            disease_name=disease_name,
            report_context=report_context,
            top_features=top_features,
        )

        messages = build_chat_messages(
            user_message=prompt,
            report_context=report_context,
            knowledge_context=self._feature_context(
                top_features
            ),
            history=None,
        )

        response = self.generate(
            messages,
            fallback_context=prompt,
        )

        return self._bullet_list(
            response,
            fallback=self._fallback_lifestyle(
                disease_name,
                top_features,
            ),
        )

    # ------------------------------------------------------------------
    # Report summary
    # ------------------------------------------------------------------

    def summarize_report(
        self,
        *,
        report_context: str,
        top_features: list[dict[str, object]],
    ) -> str:
        prompt = (
            "Summarize the uploaded report in plain language "
            "for a patient. "
            "Only use the supplied clinical context and "
            "feature details."
        )

        messages = build_chat_messages(
            user_message=prompt,
            report_context=report_context,
            knowledge_context=self._feature_context(
                top_features
            ),
            history=None,
        )

        return self.generate(
            messages,
            fallback_context=(
                f"{prompt}\n{report_context}"
            ),
        )

    # ------------------------------------------------------------------
    # Chat question
    # ------------------------------------------------------------------

    def answer_question(
        self,
        *,
        user_message: str,
        report_context: str,
        knowledge_context: str,
        history: Iterable[dict[str, str]] | None = None,
    ) -> str:
        messages = build_chat_messages(
            user_message=user_message,
            report_context=report_context,
            knowledge_context=knowledge_context,
            history=history,
        )

        fallback_context = "\n\n".join(
            [
                report_context,
                knowledge_context,
                user_message,
            ]
        )

        return self.generate(
            messages,
            fallback_context=fallback_context,
        )

    # ------------------------------------------------------------------
    # Local text streaming
    # ------------------------------------------------------------------

    def stream_text(
        self,
        text: str,
        chunk_size: int = 96,
    ):
        buffer = text.strip()

        if not buffer:
            yield ""
            return

        for start in range(
            0,
            len(buffer),
            chunk_size,
        ):
            yield buffer[
                start : start + chunk_size
            ]

    # ------------------------------------------------------------------
    # Provider generation
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[dict[str, str]],
        fallback_context: str = "",
    ) -> str:
        providers = self._provider_chain()

        last_error: Exception | None = None

        for provider in providers:
            try:
                result = self._generate_with_provider(
                    provider,
                    messages,
                )

                if result and result.strip():
                    return result.strip()

            except Exception as exc:
                # Keep trying the next provider.
                last_error = exc

        # Never leave the chat request without a response.
        return self._grounded_fallback(
            messages,
            fallback_context=fallback_context,
            error=last_error,
        )

    # ------------------------------------------------------------------
    # Provider chain
    # ------------------------------------------------------------------

    def _provider_chain(self) -> list[str]:
        preferred = self.provider

        chain = {
            "groq": [
                "groq",
                "openai",
                "ollama",
            ],
            "openai": [
                "openai",
                "groq",
                "ollama",
            ],
            "ollama": [
                "ollama",
                "groq",
                "openai",
            ],
        }

        return chain.get(
            preferred,
            [
                "groq",
                "openai",
                "ollama",
            ],
        )

    # ------------------------------------------------------------------
    # Provider dispatcher
    # ------------------------------------------------------------------

    def _generate_with_provider(
        self,
        provider: str,
        messages: list[dict[str, str]],
    ) -> str:
        if provider == "groq":
            return self._generate_with_groq(
                messages
            )

        if provider == "openai":
            return self._generate_with_openai(
                messages
            )

        if provider == "ollama":
            return self._generate_with_ollama(
                messages
            )

        raise ValueError(
            f"Unsupported provider: {provider}"
        )

    # ------------------------------------------------------------------
    # Groq
    # ------------------------------------------------------------------

    def _generate_with_groq(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not configured"
            )

        try:
            from groq import Groq
        except Exception as exc:
            raise RuntimeError(
                "Groq client is unavailable"
            ) from exc

        client = Groq(
            api_key=GROQ_API_KEY,
            timeout=self.request_timeout,
        )

        completion = client.chat.completions.create(
            model=self.groq_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=700,
            timeout=self.request_timeout,
        )

        if not completion.choices:
            raise RuntimeError(
                "Groq returned no choices"
            )

        content = (
            completion
            .choices[0]
            .message
            .content
        )

        if not content:
            raise RuntimeError(
                "Groq returned an empty response"
            )

        return content.strip()

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------

    def _generate_with_openai(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        api_key = os.getenv(
            "OPENAI_API_KEY",
            "",
        )

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured"
            )

        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError(
                "OpenAI client is unavailable"
            ) from exc

        client = OpenAI(
            api_key=api_key,
            timeout=self.request_timeout,
        )

        completion = client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=700,
            timeout=self.request_timeout,
        )

        if not completion.choices:
            raise RuntimeError(
                "OpenAI returned no choices"
            )

        content = (
            completion
            .choices[0]
            .message
            .content
        )

        if not content:
            raise RuntimeError(
                "OpenAI returned an empty response"
            )

        return content.strip()

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------

    def _generate_with_ollama(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )

        response = requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json={
                "model": self.ollama_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                },
            },
            timeout=self.request_timeout,
        )

        response.raise_for_status()

        payload = response.json()

        if isinstance(payload, dict):
            message = payload.get(
                "message"
            )

            if isinstance(
                message,
                dict,
            ):
                content = str(
                    message.get(
                        "content",
                        "",
                    )
                ).strip()

                if content:
                    return content

            if "response" in payload:
                content = str(
                    payload.get(
                        "response",
                        "",
                    )
                ).strip()

                if content:
                    return content

        raise RuntimeError(
            "Ollama returned an empty response"
        )

    # ------------------------------------------------------------------
    # Grounded fallback
    # ------------------------------------------------------------------

    def _grounded_fallback(
        self,
        messages: list[dict[str, str]],
        *,
        fallback_context: str,
        error: Exception | None,
    ) -> str:
        user_message = ""

        for message in reversed(messages):
            if message.get("role") == "user":
                user_message = message.get(
                    "content",
                    "",
                )
                break

        context_sentences = self._first_sentences(
            fallback_context,
            limit=4,
        )

        if context_sentences:
            question = self._compact_sentence(
                user_message
            )

            return (
                "Based on the available clinical context, "
                f"{question} "
                f"{' '.join(context_sentences)} "
                "Please review important findings with "
                "a licensed clinician."
            ).strip()

        return (
            "I can only answer using the uploaded report "
            "and retrieved medical references. "
            "Please upload a report or ask about a known "
            "laboratory finding."
        )

    # ------------------------------------------------------------------
    # Feature context
    # ------------------------------------------------------------------

    def _feature_context(
        self,
        top_features: list[dict[str, object]],
    ) -> str:
        if not top_features:
            return ""

        return "\n".join(
            (
                f"{item.get("
                "parameter",
                item.get("feature", "Feature")
                )}: "
                f"{item.get("
                "contribution_score",
                item.get("contribution", 0)
                )}"
            )
            for item in top_features[:5]
        )

    # ------------------------------------------------------------------
    # Lifestyle fallback
    # ------------------------------------------------------------------

    def _fallback_lifestyle(
        self,
        disease_name: str,
        top_features: list[dict[str, object]],
    ) -> list[str]:
        drivers = [
            str(
                item.get(
                    "parameter",
                    item.get(
                        "feature",
                        "feature",
                    ),
                )
            )
            for item in top_features[:3]
        ]

        base = {
            "Anemia": [
                "Include iron-rich foods with "
                "vitamin C sources.",
                "Discuss ferritin and B12 follow-up "
                "if symptoms persist.",
                "Avoid self-starting iron supplements "
                "without clinician guidance.",
            ],
            "Diabetes": [
                "Choose high-fiber meals and minimize "
                "sugary drinks.",
                "Stay physically active if your clinician "
                "says it is safe.",
                "Repeat glucose or HbA1c testing as "
                "recommended.",
            ],
            "Chronic Kidney Disease": [
                "Follow blood-pressure and hydration "
                "guidance from your clinician.",
                "Review salt intake and avoid unnecessary "
                "NSAID use.",
                "Monitor kidney markers and electrolytes "
                "on follow-up testing.",
            ],
        }.get(
            disease_name,
            [],
        )

        if drivers:
            base.insert(
                0,
                (
                    "Discuss the following signals with "
                    "your clinician: "
                    f"{', '.join(drivers)}."
                ),
            )

        return base or [
            "Review the report with a licensed clinician."
        ]

    # ------------------------------------------------------------------
    # Bullet list parser
    # ------------------------------------------------------------------

    def _bullet_list(
        self,
        text: str,
        fallback: list[str],
    ) -> list[str]:
        lines = [
            line.strip("-• \t")
            for line in text.splitlines()
            if line.strip()
        ]

        bullets = [
            line
            for line in lines
            if len(line) > 3
        ]

        return (
            bullets[:4]
            if bullets
            else fallback[:4]
        )

    # ------------------------------------------------------------------
    # Sentence helper
    # ------------------------------------------------------------------

    def _first_sentences(
        self,
        text: str,
        limit: int = 3,
    ) -> list[str]:
        sentences = [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                text or "",
            )
            if sentence.strip()
        ]

        return sentences[:limit]

    # ------------------------------------------------------------------
    # Compact sentence helper
    # ------------------------------------------------------------------

    def _compact_sentence(
        self,
        text: str,
    ) -> str:
        stripped = text.strip()

        if not stripped:
            return ""

        if len(stripped) == 1:
            return stripped

        return (
            stripped[0].upper()
            + stripped[1:]
        )
