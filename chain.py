"""Cadena LCEL con Ollama (modelos locales): salida estructurada, reintentos y ejecución async."""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache

import httpx
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from ollama import ResponseError
from pydantic import ValidationError

from schemas import TechnicalExtractionSchema

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Prompt modular: el texto y las instrucciones de formato son variables.
# No hay f-strings; todo se inyecta con las variables del template.
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "Eres un analista técnico senior. Extraes información estructurada "
    "a partir de textos técnicos (logs, descripciones de arquitectura, "
    "incidentes, etc.).\n\n"
    "{format_instructions}"
)

HUMAN_PROMPT = "Analiza el siguiente texto:\n\n{text}"

FORMAT_INSTRUCTIONS = (
    "Reglas de salida:\n"
    "- 'tecnologias': solo tecnologías mencionadas explícitamente; no inventes.\n"
    "- 'nivel_de_criticidad': exactamente uno de: baja, media, alta.\n"
    "- 'resumen_tecnico': entre 1 y 3 oraciones, en español, máximo 600 caracteres.\n"
    "- Responde únicamente con los campos del esquema solicitado."
)

DEFAULT_MODEL = "llama3.1"
DEFAULT_BASE_URL = "http://localhost:11434"

# Los modelos locales (sobre todo los chicos) fallan más seguido en el formato.
FORMAT_ERRORS: tuple[type[BaseException], ...] = (
    ValidationError,
    OutputParserException,
    ValueError,
)

# Timeouts: la primera llamada puede tardar mientras Ollama carga el modelo en memoria.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (httpx.TimeoutException,)

# Ollama no está corriendo / no es alcanzable.
CONNECTION_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    httpx.ConnectError,
)


class ProcessingError(RuntimeError):
    """Error de alto nivel cuando el pipeline no logra producir un resultado válido."""


def build_prompt() -> ChatPromptTemplate:
    """Construye el prompt reutilizable con las instrucciones de formato como partial."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )
    return prompt.partial(format_instructions=FORMAT_INSTRUCTIONS)


def build_llm() -> BaseChatModel:
    """Crea el cliente de Ollama a partir de variables de entorno."""
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
        base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
        keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "5m"),
        client_kwargs={"timeout": float(os.getenv("LLM_TIMEOUT", "120"))},
    )


def with_resilience(
    chain: Runnable[dict[str, str], TechnicalExtractionSchema],
    max_attempts: int,
) -> Runnable[dict[str, str], TechnicalExtractionSchema]:
    """Agrega reintentos con backoff exponencial y jitter.

    Reintenta fallos de formato (salida inválida/incompleta) y timeouts.
    NO reintenta si Ollama no está corriendo o el modelo no existe: ahí
    reintentar no sirve y conviene fallar rápido con un mensaje claro.
    """
    return chain.with_retry(
        retry_if_exception_type=FORMAT_ERRORS + TRANSIENT_ERRORS,
        wait_exponential_jitter=True,
        stop_after_attempt=max_attempts,
    )


def build_chain() -> Runnable[dict[str, str], TechnicalExtractionSchema]:
    """Compone la cadena LCEL: prompt | modelo con salida estructurada + reintentos."""
    structured_model = build_llm().with_structured_output(
        TechnicalExtractionSchema,
        method="json_schema",  # usa el modo de salidas estructuradas de Ollama
    )
    chain = build_prompt() | structured_model
    return with_resilience(chain, int(os.getenv("MAX_RETRIES", "3")))


@lru_cache(maxsize=1)
def get_chain() -> Runnable[dict[str, str], TechnicalExtractionSchema]:
    """Instancia única de la cadena (evita reconstruir el cliente en cada llamada)."""
    return build_chain()


async def process_text(text: str) -> TechnicalExtractionSchema:
    """Procesa un texto y devuelve el objeto Pydantic validado.

    Raises:
        ValueError: si el texto está vacío.
        ProcessingError: si Ollama no responde o la salida no es válida tras los reintentos.
    """
    if not text or not text.strip():
        raise ValueError("El texto de entrada no puede estar vacío.")

    model_name = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
    logger.info("Iniciando procesamiento (%d caracteres) con %s.", len(text), model_name)
    started = time.perf_counter()

    try:
        result = await get_chain().ainvoke({"text": text})
    except CONNECTION_ERRORS as exc:
        logger.error("No se pudo conectar con Ollama en %s: %s", base_url, exc)
        raise ProcessingError(
            f"No se pudo conectar con Ollama en {base_url}. "
            "¿Está corriendo? Probá con `ollama serve`."
        ) from exc
    except ResponseError as exc:
        if exc.status_code == 404:
            logger.error("Modelo '%s' no encontrado en Ollama.", model_name)
            raise ProcessingError(
                f"El modelo '{model_name}' no está descargado. "
                f"Ejecutá: ollama pull {model_name}"
            ) from exc
        logger.error("Error de Ollama (%s): %s", exc.status_code, exc)
        raise ProcessingError(f"Ollama devolvió un error: {exc}") from exc
    except (ValidationError, OutputParserException) as exc:
        logger.error("Salida inválida tras agotar reintentos: %s", exc)
        raise ProcessingError(
            "El modelo no devolvió una salida válida según el esquema. "
            "Probá con un modelo más grande (OLLAMA_MODEL)."
        ) from exc
    except httpx.TimeoutException as exc:
        logger.error("Timeout esperando a Ollama: %s", exc)
        raise ProcessingError(
            "Ollama tardó demasiado en responder. "
            "Aumentá LLM_TIMEOUT o usá un modelo más liviano."
        ) from exc
    except Exception as exc:  # cualquier otro fallo inesperado
        logger.exception("Fallo inesperado ejecutando la cadena.")
        raise ProcessingError(f"Fallo ejecutando la cadena: {exc}") from exc

    elapsed = time.perf_counter() - started
    logger.info(
        "Procesamiento OK en %.2fs | criticidad=%s | tecnologías=%d",
        elapsed,
        result.nivel_de_criticidad.value,
        len(result.tecnologias),
    )
    return result
