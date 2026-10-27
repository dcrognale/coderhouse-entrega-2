"""Esquemas Pydantic que definen el contrato de salida del pipeline."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NivelCriticidad(str, Enum):
    """Niveles de criticidad permitidos."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class TechnicalExtractionSchema(BaseModel):
    """Resultado estructurado de la extracción técnica sobre un texto.

    El esquema JSON enviado al modelo se mantiene simple; las reglas más
    estrictas (lista sin duplicados ni vacíos, largo del resumen) se validan
    localmente con Pydantic, y si fallan se dispara el reintento de la cadena.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    tecnologias: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Tecnologías, lenguajes, frameworks, servicios o herramientas "
            "mencionados explícitamente en el texto. Al menos una."
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        ...,
        description=(
            "Criticidad del problema o sistema descrito: "
            "'baja', 'media' o 'alta'."
        ),
    )
    resumen_tecnico: str = Field(
        ...,
        description="Resumen técnico conciso (1 a 3 oraciones) del texto.",
    )

    @field_validator("tecnologias")
    @classmethod
    def _clean_tecnologias(cls, value: list[str]) -> list[str]:
        """Normaliza la lista: quita vacíos y duplicados (sin alterar el orden)."""
        seen: set[str] = set()
        cleaned: list[str] = []
        for item in value:
            name = item.strip()
            key = name.lower()
            if name and key not in seen:
                seen.add(key)
                cleaned.append(name)

        if not cleaned:
            raise ValueError("La lista 'tecnologias' no puede estar vacía.")
        return cleaned

    @field_validator("resumen_tecnico")
    @classmethod
    def _validate_resumen(cls, value: str) -> str:
        """Valida el largo del resumen (se hace acá y no en el JSON Schema)."""
        if not 10 <= len(value) <= 600:
            raise ValueError(
                "El 'resumen_tecnico' debe tener entre 10 y 600 caracteres."
            )
        return value
