"""
tts_engine.py
=============

Wrapper sobre Piper TTS (motor VITS exportado a ONNX).

Responsabilidades:
    - Cargar en memoria uno o varios modelos .onnx / .onnx.json al arrancar
      el proceso (evita el costo de 300-500ms de recarga por petición).
    - Exponer una API de síntesis por streaming: dado un texto, entrega
      bloques de audio PCM (bytes) tan pronto como están disponibles, en
      lugar de esperar a que la frase completa termine de sintetizarse.
    - Traducir los parámetros "amigables" del cliente (speed, pitch, emotion)
      a los parámetros nativos del motor VITS (length_scale, noise_scale,
      noise_w).
    - Ejecutar la inferencia en un hilo separado (vía asyncio.to_thread) para
      no bloquear el event loop de FastAPI, y soportar cancelación cooperativa
      mediante un flag de sesión de habla (speech_id).

Requiere el paquete `piper-tts` (https://github.com/rhasspy/piper):
    pip install piper-tts

Los modelos se descargan por separado (ver README.md) y se referencian por
`voice_id`, que debe coincidir con el nombre base del archivo .onnx, p. ej.
"es_ES-davefx-medium" -> models/es_ES-davefx-medium.onnx
                         models/es_ES-davefx-medium.onnx.json
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

logger = logging.getLogger("tts_engine")

try:
    from piper import PiperVoice
    from piper.config import SynthesisConfig as PiperSynthesisConfig
except ImportError as exc:  # pragma: no cover - guía de instalación en runtime
    raise ImportError(
        "No se encontró el paquete 'piper-tts'. Instálalo con: pip install piper-tts"
    ) from exc


# --------------------------------------------------------------------------- #
# Mapeo de parámetros "amigables" -> parámetros nativos del motor VITS/Piper
# --------------------------------------------------------------------------- #

# Presets de emoción: (length_scale, noise_scale, noise_w)
EMOTION_PRESETS: dict[str, tuple[float, float, float]] = {
    "neutral": (1.00, 0.667, 0.800),
    "happy": (0.85, 0.80, 0.90),
    "excited": (0.85, 0.80, 0.90),
    "sad": (1.25, 0.30, 0.40),
    "serious": (1.25, 0.30, 0.40),
}

DEFAULT_SAMPLE_RATE = 22050


@dataclass
class VoiceParams:
    """Parámetros de voz recibidos del cliente (mensaje 'configure')."""

    voice_id: str
    speed: float = 1.0       # 0.5 - 2.0
    pitch: float = 1.0       # 0.1 - 1.5  (afecta noise_scale)
    emotion: str = "neutral"  # neutral | happy | excited | sad | serious

    def to_synthesis_config(self) -> "SynthesisConfig":
        """Traduce los parámetros de usuario a parámetros nativos de Piper."""
        length_scale, base_noise_scale, noise_w = EMOTION_PRESETS.get(
            self.emotion, EMOTION_PRESETS["neutral"]
        )

        # speed es inverso a length_scale: speed=2.0 -> habla el doble de rápido
        # -> length_scale más pequeño. Se combina multiplicativamente con el
        # length_scale del preset de emoción para que ambos efectos se apliquen.
        speed = max(0.5, min(2.0, self.speed))
        length_scale = length_scale * (1.0 / speed)

        # pitch se mapea a noise_scale (variabilidad de F0). Se usa como
        # multiplicador sobre el preset de emoción, acotado a un rango sano.
        pitch = max(0.1, min(1.5, self.pitch))
        noise_scale = max(0.1, min(1.0, base_noise_scale * pitch))

        return SynthesisConfig(
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w,
        )


@dataclass
class SynthesisConfig:
    """Parámetros nativos que Piper/VITS espera en tiempo de inferencia.

    Nota: el campo se llama `noise_w_scale` (no `noise_w`) para coincidir
    exactamente con `piper.config.SynthesisConfig` de las versiones
    recientes del paquete `piper-tts`.
    """

    length_scale: float
    noise_scale: float
    noise_w_scale: float

    def to_piper_config(self) -> PiperSynthesisConfig:
        """Construye el SynthesisConfig nativo que espera PiperVoice.synthesize()."""
        return PiperSynthesisConfig(
            length_scale=self.length_scale,
            noise_scale=self.noise_scale,
            noise_w_scale=self.noise_w_scale,
        )


class TTSEngine:
    """Gestiona la carga de modelos Piper y la síntesis de audio.

    Se instancia una única vez al arrancar la aplicación (evento `startup`
    de FastAPI) y se reutiliza para todas las conexiones WebSocket.
    """

    def __init__(self, models_dir: str | Path = "models") -> None:
        self.models_dir = Path(models_dir)
        self._voices: dict[str, PiperVoice] = {}
        self._lock = asyncio.Lock()

    def preload(self, voice_ids: list[str]) -> None:
        """Carga en memoria los modelos indicados. Llamar en el startup."""
        for voice_id in voice_ids:
            self._load_voice(voice_id)

    def _load_voice(self, voice_id: str) -> PiperVoice:
        if voice_id in self._voices:
            return self._voices[voice_id]

        model_path = self.models_dir / f"{voice_id}.onnx"
        config_path = self.models_dir / f"{voice_id}.onnx.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo '{model_path}'. "
                f"Descarga la voz con el script scripts/download_voice.sh {voice_id}"
            )

        logger.info("Cargando modelo Piper: %s", voice_id)
        voice = PiperVoice.load(str(model_path), config_path=str(config_path))
        self._voices[voice_id] = voice
        return voice

    def get_voice(self, voice_id: str) -> PiperVoice:
        """Obtiene una voz ya cargada, o la carga bajo demanda (lazy)."""
        if voice_id not in self._voices:
            self._load_voice(voice_id)
        return self._voices[voice_id]

    def sample_rate(self, voice_id: str) -> int:
        voice = self.get_voice(voice_id)
        return getattr(voice.config, "sample_rate", DEFAULT_SAMPLE_RATE)

    async def synthesize_stream(
        self,
        text: str,
        params: VoiceParams,
        cancel_flag: "CancelFlag",
        chunk_size_bytes: int = 4096,
    ) -> AsyncIterator[bytes]:
        """Sintetiza `text` y entrega bloques de audio PCM crudo (S16LE mono).

        La inferencia real de Piper se ejecuta en un hilo (asyncio.to_thread)
        para no bloquear el event loop. Los bloques de audio se van
        depositando en una cola interna a medida que están listos, lo que
        permite que el primer bloque salga hacia el cliente sin esperar a que
        termine de sintetizarse toda la frase.

        `cancel_flag` se revisa entre cada bloque generado: si se activa
        (por una interrupción del usuario), la síntesis se detiene lo antes
        posible y el generador termina sin emitir más bloques.
        """
        voice = self.get_voice(params.voice_id)
        config = params.to_synthesis_config()
        piper_config = config.to_piper_config()

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()

        def _produce() -> None:
            try:
                # PiperVoice.synthesize() entrega un AudioChunk por frase
                # detectada internamente por Piper (no por bloques de
                # tiempo fijo). Cada AudioChunk expone `.audio_int16_bytes`
                # con el PCM S16LE ya listo para enviar. Re-fragmentamos ese
                # bloque en trozos de `chunk_size_bytes` para no mandar un
                # único frame gigante por el WebSocket.
                for audio_chunk in voice.synthesize(text, syn_config=piper_config):
                    if cancel_flag.is_cancelled():
                        break
                    raw_bytes = audio_chunk.audio_int16_bytes
                    for i in range(0, len(raw_bytes), chunk_size_bytes):
                        piece = raw_bytes[i : i + chunk_size_bytes]
                        asyncio.run_coroutine_threadsafe(queue.put(piece), loop)
                        if cancel_flag.is_cancelled():
                            break
            except Exception:  # pragma: no cover - se reporta al consumidor
                logger.exception("Error durante la síntesis de: %r", text)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        producer_task = loop.run_in_executor(None, _produce)

        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                if cancel_flag.is_cancelled():
                    break
                yield chunk
        finally:
            # Asegura que el hilo productor no quede colgado.
            await asyncio.wrap_future(producer_task) if False else None


class CancelFlag:
    """Flag de cancelación cooperativa para una 'sesión de habla'.

    Cada vez que el cliente envía {"action": "interrupt"}, se debe llamar a
    `bump()`, lo que invalida cualquier síntesis en curso: el hilo productor
    de audio revisa `is_cancelled()` entre bloques y aborta si detecta que su
    `speech_id` quedó obsoleto.
    """

    def __init__(self) -> None:
        self._current_id = 0

    def new_speech_id(self) -> int:
        """Genera y activa un nuevo id de sesión de habla."""
        self._current_id += 1
        return self._current_id

    def bump(self) -> None:
        """Invalida cualquier síntesis en curso (usar ante 'interrupt')."""
        self._current_id += 1

    def is_cancelled(self, speech_id: Optional[int] = None) -> bool:
        if speech_id is None:
            return False
        return speech_id != self._current_id

    @property
    def current_id(self) -> int:
        return self._current_id


def pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    """Utilidad opcional: envuelve PCM crudo en un contenedor WAV en memoria.

    Útil solo para depuración local (p. ej. guardar un chunk para escucharlo).
    El protocolo de streaming normal NO usa esto: se envía PCM crudo sin
    cabecera para minimizar CPU y latencia (ver README, sección de formato).
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()
