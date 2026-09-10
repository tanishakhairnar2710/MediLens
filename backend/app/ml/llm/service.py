from __future__ import annotations

import os
import re
from typing import Iterable

import requests

from backend.app.config.settings import GROQ_API_KEY
from backend.app.services.settings_service import get_settings
from backend.app.ml.prompts.templates import (
    build_chat_messages,
    build_lifestyle_prompt,
    build_report_explanation_prompt,
)


class LLMService:
    """
    Central LLM service for MediLens.

    Provider priority:
        1. Groq
        2. OpenAI, if configured
        3. Ollama, if explicitly configured
        4. Grounded local fallback

    The fallback keeps the application usable if an external
    LLM provider is temporarily unavailable.
    """

    def __init__(self, provider: str | None = None) -> None:
        settings = get_settings()

        self.provider = (
            provider
            or settings.get("ai_provider")
            or "groq"
        ).lower()

        # Current Groq production model.
        # Can be overridden with MEDILENS_GROQ_MODEL.
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
        """
        Streams an already-generated response in small chunks.

        The model is called only once. This preserves the existing
        frontend streaming interface without making a second LLM call.
        """

        buffer = (text or "").strip()

        if not buffer:
            yield ""
            return

        for start in range(
            0,
            len(buffer),
            chunk_size,
        ):
            yield buffer[
                start:start + chunk_size
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
                print(
                    f"[LLM] Trying provider={provider}",
                    flush=True,
                )

                result = self._generate_with_provider(
                    provider,
                    messages,
                )

                if result and result.strip():
                    print(
                        f"[LLM] Provider={provider} succeeded",
                        flush=True,
                    )

                    return result.strip()

                raise RuntimeError(
                    f"{provider} returned an empty response"
                )

            except Exception as exc:
                last_error = exc

                print(
                    f"[LLM] Provider '{provider}' failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        print(
            "[LLM] All configured providers failed. "
            "Using grounded fallback. "
            f"Last error: "
            f"{type(last_error).__name__ if last_error else 'None'}: "
            f"{last_error if last_error else 'None'}",
            flush=True,
        )

        return self._grounded_fallback(
            messages,
            fallback_context=fallback_context,
            error=last_error,
        )

    # ------------------------------------------------------------------
    # Provider chain
    # ------------------------------------------------------------------

    def _provider_chain(self) -> list[str]:
        """
        Only use providers that are actually configured.

        On Render, Ollama is skipped unless OLLAMA_BASE_URL exists.
        """

        preferred = self.provider

        groq_available = bool(
            GROQ_API_KEY
        )

        openai_available = bool(
            os.getenv(
                "OPENAI_API_KEY",
                "",
            ).strip()
        )

        ollama_available = bool(
            os.getenv(
                "OLLAMA_BASE_URL",
                "",
            ).strip()
        )

        available = {
            "groq": groq_available,
            "openai": openai_available,
            "ollama": ollama_available,
        }

        order = {
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

        candidates = order.get(
            preferred,
            [
                "groq",
                "openai",
                "ollama",
            ],
        )

        configured = [
            provider
            for provider in candidates
            if available.get(
                provider,
                False,
            )
        ]

        # Keep Groq in the chain even when the key is missing so
        # the Render log clearly reports the configuration problem.
        if not configured:
            return ["groq"]

        return configured

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

        print(
            f"[LLM] Calling Groq model="
            f"{self.groq_model}",
            flush=True,
        )

        client = Groq(
            api_key=GROQ_API_KEY,
            timeout=self.request_timeout,
        )

        completion = client.chat.completions.create(
            model=self.groq_model,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=700,
            reasoning_effort="low",
            include_reasoning=False,
        )

        if not completion.choices:
            raise RuntimeError(
                "Groq returned no choices"
            )

        message = completion.choices[0].message

        content = getattr(
            message,
            "content",
            None,
        )

        if not content:
            raise RuntimeError(
                "Groq returned an empty response"
            )

        return str(content).strip()

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
        ).strip()

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
            max_completion_tokens=700,
        )

        if not completion.choices:
            raise RuntimeError(
                "OpenAI returned no choices"
            )

        message = completion.choices[0].message

        content = getattr(
            message,
            "content",
            None,
        )

        if not content:
            raise RuntimeError(
                "OpenAI returned an empty response"
            )

        return str(content).strip()

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------

    def _generate_with_ollama(
        self,
        messages: list[dict[str, str]],
    ) -> str:

        base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "",
        ).strip()

        if not base_url:
            raise RuntimeError(
                "OLLAMA_BASE_URL is not configured"
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
        """
        Safe fallback when external LLM providers fail.

        The fallback deliberately avoids dumping the entire report
        into the chat response.
        """

        user_message = ""

        for message in reversed(messages):
            if message.get("role") == "user":
                user_message = str(
                    message.get(
                        "content",
                        "",
                    )
                ).strip()
                break

        if not user_message:
            user_message = "your question"

        answer = self._answer_common_lab_question(
            user_message=user_message,
            context=fallback_context,
        )

        if answer:
            return answer

        context_sentences = self._first_sentences(
            fallback_context,
            limit=2,
        )

        if context_sentences:
            return (
                "I can use the uploaded report as context, "
                "but the AI language model is temporarily "
                "unavailable. Please try the question again "
                "shortly."
            )

        return (
            "The AI language model is temporarily unavailable. "
            "Please try again shortly."
        )

    # ------------------------------------------------------------------
    # Simple grounded laboratory fallback
    # ------------------------------------------------------------------

    def _answer_common_lab_question(
        self,
        *,
        user_message: str,
        context: str,
    ) -> str:

        question = (
            user_message or ""
        ).lower()

        report = context or ""

        # --------------------------------------------------------------
        # Hemoglobin
        # --------------------------------------------------------------

        if (
            "hemoglobin" in question
            or "haemoglobin" in question
        ):
            match = re.search(
                r"hemoglobin\s+"
                r"([0-9]+(?:\.[0-9]+)?)"
                r"\s*g/dl"
                r"(?:\s+"
                r"([0-9]+(?:\.[0-9]+)?)"
                r"[–-]"
                r"([0-9]+(?:\.[0-9]+)?))?",
                report,
                flags=re.IGNORECASE,
            )

            if match:
                value = float(
                    match.group(1)
                )

                low = (
                    float(match.group(2))
                    if match.group(2)
                    else None
                )

                high = (
                    float(match.group(3))
                    if match.group(3)
                    else None
                )

                if (
                    low is not None
                    and high is not None
                    and low <= value <= high
                ):
                    return (
                        f"The report shows hemoglobin at "
                        f"{value:g} g/dL. This is within "
                        f"the report's reference range of "
                        f"{low:g}–{high:g} g/dL. "
                        "Hemoglobin is a protein in red blood "
                        "cells that carries oxygen."
                    )

                return (
                    f"The report shows hemoglobin at "
                    f"{value:g} g/dL. Whether that is low, "
                    "normal, or high should be determined "
                    "using the reference range shown on "
                    "the report."
                )

            return (
                "Hemoglobin is a protein in red blood cells "
                "that carries oxygen around the body. "
                "Its result should be interpreted using the "
                "reference range shown on the report."
            )

        # --------------------------------------------------------------
        # MCV
        # --------------------------------------------------------------

        if "mcv" in question:
            match = re.search(
                r"MCV\s+"
                r"([0-9]+(?:\.[0-9]+)?)"
                r"\s*fL"
                r"(?:\s+"
                r"([0-9]+(?:\.[0-9]+)?)"
                r"[–-]"
                r"([0-9]+(?:\.[0-9]+)?))?",
                report,
                flags=re.IGNORECASE,
            )

            if match:
                value = float(
                    match.group(1)
                )

                low = (
                    float(match.group(2))
                    if match.group(2)
                    else None
                )

                high = (
                    float(match.group(3))
                    if match.group(3)
                    else None
                )

                if (
                    low is not None
                    and high is not None
                    and low <= value <= high
                ):
                    return (
                        f"The report shows an MCV of "
                        f"{value:g} fL, which is within "
                        f"the report's reference range of "
                        f"{low:g}–{high:g} fL. "
                        "MCV describes the average size "
                        "of red blood cells."
                    )

            return (
                "MCV describes the average size of red "
                "blood cells. The reference range on the "
                "report should be used to determine "
                "whether it is low, normal, or high."
            )

        # --------------------------------------------------------------
        # Cholesterol
        # --------------------------------------------------------------

        if (
            "cholesterol" in question
            or "ldl" in question
            or "hdl" in question
        ):
            return (
                "The report includes total cholesterol, LDL, "
                "HDL, and triglycerides. These measurements "
                "describe different aspects of blood lipid "
                "levels. Their results should be interpreted "
                "using the reference or target ranges shown "
                "on the report."
            )

        # --------------------------------------------------------------
        # Glucose / diabetes
        # --------------------------------------------------------------

        if (
            "glucose" in question
            or "blood sugar" in question
            or "diabetes" in question
        ):
            return (
                "The report includes a fasting glucose value. "
                "Fasting glucose measures blood sugar after "
                "fasting and is one factor clinicians use "
                "when assessing glucose regulation."
            )

        # --------------------------------------------------------------
        # Creatinine / kidney
        # --------------------------------------------------------------

        if (
            "creatinine" in question
            or "kidney" in question
        ):
            return (
                "Creatinine is a blood measurement commonly "
                "used with other information to assess kidney "
                "function. The result should be interpreted "
                "using the report's reference range and the "
                "person's clinical context."
            )

        # --------------------------------------------------------------
        # WBC
        # --------------------------------------------------------------

        if (
            "wbc" in question
            or "white blood" in question
            or "white cell" in question
        ):
            return (
                "WBC stands for white blood cell count. "
                "White blood cells are part of the immune "
                "system. The report's reference range is "
                "used to determine whether the count is "
                "within the stated range."
            )

        # --------------------------------------------------------------
        # Platelets
        # --------------------------------------------------------------

        if "platelet" in question:
            return (
                "Platelets are blood components involved "
                "in normal blood clotting. Their result "
                "should be interpreted against the reference "
                "range shown on the report."
            )

        return ""

    # ------------------------------------------------------------------
    # Feature context
    # ------------------------------------------------------------------

    def _feature_context(
        self,
        top_features: list[dict[str, object]],
    ) -> str:

        if not top_features:
            return ""

        lines: list[str] = []

        for item in top_features[:5]:

            parameter = item.get(
                "parameter",
                item.get(
                    "feature",
                    "Feature",
                ),
            )

            contribution = item.get(
                "contribution_score",
                item.get(
                    "contribution",
                    0,
                ),
            )

            lines.append(
                f"{parameter}: {contribution}"
            )

        return "\n".join(lines)

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
                "Include iron-rich foods with vitamin C sources.",
                "Discuss ferritin and B12 follow-up if symptoms persist.",
                "Avoid self-starting iron supplements without clinician guidance.",
            ],
            "Diabetes": [
                "Choose high-fiber meals and minimize sugary drinks.",
                "Stay physically active if your clinician says it is safe.",
                "Repeat glucose or HbA1c testing as recommended.",
            ],
            "Chronic Kidney Disease": [
                "Follow blood-pressure and hydration guidance from your clinician.",
                "Review salt intake and avoid unnecessary NSAID use.",
                "Monitor kidney markers and electrolytes on follow-up testing.",
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
            for line in (
                text or ""
            ).splitlines()
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

        stripped = (
            text or ""
        ).strip()

        if not stripped:
            return ""

        if len(stripped) == 1:
            return stripped

        return (
            stripped[0].upper()
            + stripped[1:]
        )
