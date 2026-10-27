# Pre-entrega 2: Pipeline de procesamiento validado (Ollama)

Pipeline asíncrono con LangChain (LCEL) y **Ollama** (modelos locales) que extrae
información técnica estructurada de un texto y garantiza la salida con un esquema Pydantic.
No requiere API keys ni envía datos a servicios externos.

## Arquitectura

```
texto → ChatPromptTemplate → ChatOllama.with_structured_output(Schema) → .with_retry() → objeto Pydantic
```

- `schemas.py`: modelo `TechnicalExtractionSchema` (lista no vacía, enum de criticidad, resumen validado).
- `chain.py`: prompt modular, cadena LCEL con Ollama, reintentos y `process_text()` async con logs.
- `main.py`: script de prueba ejecutable.

## Configuración del proyecto

1. Crear y activar un entorno virtual (Python 3.10+):

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

2. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

3. Crear el archivo `.env` (los valores por defecto sirven si Ollama corre en tu máquina):

   ```bash
   cp .env.example .env
   ```

   | Variable | Descripción | Default |
   |---|---|---|
   | `OLLAMA_BASE_URL` | URL del servidor de Ollama | `http://localhost:11434` |
   | `OLLAMA_MODEL` | Modelo a usar (debe estar descargado) | `llama3.1` |
   | `OLLAMA_NUM_CTX` | Ventana de contexto en tokens | `8192` |
   | `OLLAMA_KEEP_ALIVE` | Tiempo que el modelo queda cargado en memoria | `5m` |
   | `LLM_TEMPERATURE` | Temperatura | `0` |
   | `LLM_TIMEOUT` | Timeout en segundos (la primera llamada incluye la carga del modelo) | `120` |
   | `MAX_RETRIES` | Intentos máximos de `.with_retry()` | `3` |

## Ejecución

```bash
python main.py
```

## Ejemplo de salida esperada

```json
{
  "tecnologias": [
    "PostgreSQL",
    "Supabase",
    "Kafka",
    "Stripe",
    "Spring Boot 3",
    "Railway",
    "HikariCP"
  ],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "El servicio de pagos agotó el pool de conexiones a PostgreSQL y acumula lag en Kafka, provocando reintentos y eventos duplicados de webhooks de Stripe. Los clientes no pueden completar el checkout."
}
```

Los valores exactos varían según el modelo que uses.

## Resiliencia

- `.with_retry()` reintenta con backoff exponencial y jitter ante fallos de formato
  (`ValidationError`, `OutputParserException`, `ValueError`) y timeouts. Los modelos locales
  fallan más seguido el formato que los modelos en la nube, por eso este paso es importante.
- Si Ollama no está corriendo o el modelo no está descargado, **no se reintenta**: falla rápido
  con un mensaje que indica cómo solucionarlo (`ollama serve` / `ollama pull <modelo>`).
- Si se agotan los intentos, `process_text()` lanza `ProcessingError`.

## Problemas comunes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `No se pudo conectar con Ollama` | El servidor no está corriendo | `ollama serve` |
| `El modelo '...' no está descargado` | Falta el modelo | `ollama pull <modelo>` |
| Timeout en la primera ejecución | El modelo se está cargando en memoria | Subir `LLM_TIMEOUT` |
| `ProcessingError` por salida inválida | Modelo demasiado chico | Usar `llama3.1` o `qwen2.5` (7B–8B) |
