"""Script de prueba: ejecuta el pipeline sobre un texto de ejemplo."""

from __future__ import annotations

import asyncio
import logging
import sys

from dotenv import load_dotenv

load_dotenv()  # debe cargarse antes de instanciar el cliente LLM

from chain import ProcessingError, process_text
from schemas import TechnicalExtractionSchema

SAMPLE_TEXT = """
2026-10-26 03:14:07 ERROR [payment-service] Connection pool exhausted:
HikariPool-1 - Timeout failure after 30000ms. PostgreSQL (Supabase) reports
max_connections=60 reached. Kafka consumer group 'payments-webhooks' lag is
184k messages and growing. Stripe webhooks are being retried and duplicated
events are reaching the Spring Boot 3 service deployed on Railway.
Customers are unable to complete checkout.
"""


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


async def run() -> int:
    try:
        result: TechnicalExtractionSchema = await process_text(SAMPLE_TEXT)
    except (ValueError, ProcessingError) as exc:
        logging.getLogger(__name__).error("Pipeline falló: %s", exc)
        return 1

    print("\n=== Resultado validado ===")
    print(result.model_dump_json(indent=2))
    print(f"\nTipo: {type(result).__name__}")
    return 0


if __name__ == "__main__":
    configure_logging()
    sys.exit(asyncio.run(run()))
