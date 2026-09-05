"""
main.py
=======

API de streaming de voz (Text-to-Speech) de baja latencia sobre FastAPI +
WebSockets, pensada para ejecutarse en servidores con CPU (sin GPU dedicada).

Endpoint principal: ws://<host>:<port>/ws/v1/audio-stream

Ciclo de vida de una sesión:
    1. El cliente abre el WebSocket y envía un mensaje `configure` con los
       parámetros de voz (voice_id, speed, pitch, emotion).
    2. El cliente envía uno o más mensajes `text_chunk` a medida que el LLM
       genera texto. El servidor los acumula, los fragmenta en frases
       (text_buffer.SentenceBuffer) y sintetiza cada frase en cuanto está
       lista, sin esperar al resto del texto.
    3. El servidor devuelve, intercalados:
         - Frames de texto JSON: {"event": "sentence_start"/"sentence_end", "id": N}
         - Frames binarios: bloques de audio PCM (16-bit, mono, ver sample_rate
           informado en la respuesta de "configure_ack").
    4. El cliente puede enviar `interrupt` en cualquier momento para cancelar
       la síntesis en curso y vaciar los buffers pendientes.
    5. El cliente puede enviar `end_of_turn` para indicar que no llegará más
       texto en este turno (fuerza el flush del buffer de frase).

Ejecutar con:
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

Ver README.md para más detalles del protocolo y de despliegue.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, ValidationError

from text_buffer import SentenceBuffer
from tts_engine import CancelFlag, TTSEngine, VoiceParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("main")

# --------------------------------------------------------------------------- #
# Configuración de la aplicación
# --------------------------------------------------------------------------- #

# Voces a precargar en memoria al arrancar el proceso. Ajustar según los
# modelos .onnx disponibles en la carpeta `models/`.
PRELOAD_VOICES = ["es_ES-davefx-medium"]

AUDIO_CHUNK_SIZE_BYTES = 4096  # tamaño de cada frame binario enviado al cliente

tts_engine = TTSEngine(models_dir="models")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Precargando modelos de voz: %s", PRELOAD_VOICES)
    try:
        tts_engine.preload(PRELOAD_VOICES)
    except FileNotFoundError as exc:
        # No detenemos el arranque: permite iniciar el servidor para pruebas
        # de protocolo aunque el modelo aún no se haya descargado, pero se
        # avisa claramente en el log.
        logger.warning("No se pudieron precargar todas las voces: %s", exc)
    yield
    logger.info("Cerrando aplicación TTS.")


app = FastAPI(
    title="Voice Streaming API",
    description="API de streaming de voz (TTS) de baja latencia sobre WebSockets.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Modelos de mensajes del protocolo (validación con Pydantic)
# --------------------------------------------------------------------------- #

class ConfigureVoiceParams(BaseModel):
    voice_id: str
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    pitch: float = Field(default=1.0, ge=0.1, le=1.5)
    emotion: str = Field(default="neutral")


class ConfigureMessage(BaseModel):
    action: str  # "configure"
    voice_params: ConfigureVoiceParams


class TextChunkMessage(BaseModel):
    action: str  # "text_chunk"
    text: str


class SimpleActionMessage(BaseModel):
    action: str  # "interrupt" | "end_of_turn"


# --------------------------------------------------------------------------- #
# Estado por sesión (una instancia por conexión WebSocket)
# --------------------------------------------------------------------------- #

class SessionState:
    """Encapsula todo el estado mutable de una conexión WebSocket activa."""

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.voice_params: Optional[VoiceParams] = None
        self.sentence_buffer = SentenceBuffer()
        self.cancel_flag = CancelFlag()
        self.text_queue: asyncio.Queue[Optional[tuple[int, str]]] = asyncio.Queue()
        self.next_sentence_id = 0
        self._send_lock = asyncio.Lock()

    def is_configured(self) -> bool:
        return self.voice_params is not None

    async def send_json(self, payload: dict) -> None:
        async with self._send_lock:
            await self.websocket.send_text(json.dumps(payload, ensure_ascii=False))

    async def send_bytes(self, data: bytes) -> None:
        async with self._send_lock:
            await self.websocket.send_bytes(data)


# --------------------------------------------------------------------------- #
# Tareas asíncronas: consumidor de texto -> síntesis -> audio
# --------------------------------------------------------------------------- #

async def synthesis_worker(state: SessionState) -> None:
    """Consume `state.text_queue` y sintetiza cada frase, emitiendo audio.

    Corre como una tarea de fondo por sesión (asyncio.create_task), en
    paralelo con el receptor de mensajes del WebSocket (asyncio.gather).
    """
    while True:
        item = await state.text_queue.get()
        if item is None:  # señal de cierre de la sesión
            break

        sentence_id, sentence_text = item
        my_speech_id = state.cancel_flag.new_speech_id()

        try:
            await state.send_json({"event": "sentence_start", "id": sentence_id})

            assert state.voice_params is not None
            async for audio_chunk in tts_engine.synthesize_stream(
                sentence_text,
                state.voice_params,
                cancel_flag=state.cancel_flag,
                chunk_size_bytes=AUDIO_CHUNK_SIZE_BYTES,
            ):
                if state.cancel_flag.is_cancelled(my_speech_id):
                    break
                await state.send_bytes(audio_chunk)

            if not state.cancel_flag.is_cancelled(my_speech_id):
                await state.send_json({"event": "sentence_end", "id": sentence_id})

        except WebSocketDisconnect:
            break
        except Exception:
            logger.exception("Error sintetizando la frase %s", sentence_id)
            await state.send_json(
                {"event": "error", "id": sentence_id, "message": "synthesis_failed"}
            )


# --------------------------------------------------------------------------- #
# Endpoint WebSocket
# --------------------------------------------------------------------------- #

@app.websocket("/ws/v1/audio-stream")
async def audio_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    state = SessionState(websocket)
    worker_task = asyncio.create_task(synthesis_worker(state))

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                payload = json.loads(raw_message)
            except json.JSONDecodeError:
                await state.send_json({"event": "error", "message": "invalid_json"})
                continue

            action = payload.get("action")

            if action == "configure":
                await _handle_configure(state, payload)
            elif action == "text_chunk":
                await _handle_text_chunk(state, payload)
            elif action == "interrupt":
                await _handle_interrupt(state)
            elif action == "end_of_turn":
                await _handle_end_of_turn(state)
            else:
                await state.send_json(
                    {"event": "error", "message": f"unknown_action: {action}"}
                )

    except WebSocketDisconnect:
        logger.info("Cliente desconectado.")
    finally:
        state.cancel_flag.bump()
        await state.text_queue.put(None)
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


async def _handle_configure(state: SessionState, payload: dict) -> None:
    try:
        msg = ConfigureMessage.model_validate(payload)
    except ValidationError as exc:
        await state.send_json({"event": "error", "message": f"invalid_configure: {exc}"})
        return

    state.voice_params = VoiceParams(**msg.voice_params.model_dump())

    try:
        sample_rate = tts_engine.sample_rate(state.voice_params.voice_id)
    except FileNotFoundError as exc:
        await state.send_json({"event": "error", "message": str(exc)})
        state.voice_params = None
        return

    await state.send_json(
        {
            "event": "configure_ack",
            "sample_rate": sample_rate,
            "audio_format": "pcm_s16le_mono",
        }
    )


async def _handle_text_chunk(state: SessionState, payload: dict) -> None:
    if not state.is_configured():
        await state.send_json(
            {"event": "error", "message": "not_configured: envía 'configure' primero"}
        )
        return

    try:
        msg = TextChunkMessage.model_validate(payload)
    except ValidationError as exc:
        await state.send_json({"event": "error", "message": f"invalid_text_chunk: {exc}"})
        return

    for sentence in state.sentence_buffer.feed(msg.text):
        state.next_sentence_id += 1
        await state.text_queue.put((state.next_sentence_id, sentence))


async def _handle_interrupt(state: SessionState) -> None:
    # 1. Invalida cualquier síntesis en curso (cooperativo, revisado por el
    #    hilo de Piper y por synthesis_worker entre bloques).
    state.cancel_flag.bump()

    # 2. Vacía la cola de texto pendiente.
    while not state.text_queue.empty():
        try:
            state.text_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

    # 3. Descarta cualquier texto acumulado sin cortar todavía.
    state.sentence_buffer.reset()

    # 4. Confirma al cliente para que sincronice su reproductor local.
    await state.send_json({"event": "interrupted"})


async def _handle_end_of_turn(state: SessionState) -> None:
    for sentence in state.sentence_buffer.flush():
        state.next_sentence_id += 1
        await state.text_queue.put((state.next_sentence_id, sentence))
    await state.send_json({"event": "turn_complete"})


# --------------------------------------------------------------------------- #
# Endpoints auxiliares (salud del servicio / metadatos)
# --------------------------------------------------------------------------- #

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "preloaded_voices": PRELOAD_VOICES}


@app.get("/")
async def root() -> dict:
    return {
        "name": "Voice Streaming API",
        "websocket_endpoint": "/ws/v1/audio-stream",
        "docs": "/docs",
    }
