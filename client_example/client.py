"""
client.py
=========

Cliente de ejemplo en Python para la Voice Streaming API.

Ejercita el protocolo completo:
    - configure          (handshake inicial con parámetros de voz)
    - text_chunk          (envío progresivo de texto, simulando un LLM)
    - eventos de control  (sentence_start / sentence_end / configure_ack)
    - recepción de audio  (frames binarios PCM) -> se guardan en un .wav
    - interrupt           (cancelación a mitad de una síntesis)
    - end_of_turn         (fuerza el flush de la última frase pendiente)

Requiere:
    pip install websockets

Uso:
    python client.py --host localhost --port 8000 --voice es_ES-davefx-medium
    python client.py --demo interrupt      # demuestra la cancelación
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import websockets


async def run_basic_demo(uri: str, voice_id: str, out_path: Path) -> None:
    """Envía un texto completo por chunks y guarda el audio resultante."""
    async with websockets.connect(uri, max_size=None) as ws:
        # 1. Handshake / configuración de voz.
        await ws.send(
            json.dumps(
                {
                    "action": "configure",
                    "voice_params": {
                        "voice_id": voice_id,
                        "speed": 1.1,
                        "pitch": 1.0,
                        "emotion": "happy",
                    },
                }
            )
        )
        ack = json.loads(await ws.recv())
        print("[configure_ack]", ack)
        sample_rate = ack.get("sample_rate", 22050)

        # 2. Simula un LLM que va generando texto palabra por palabra.
        texto_completo = (
            "Hola, esto es una prueba del sistema de voz en streaming. "
            "¿Verdad que suena natural? Vamos a ver cómo maneja las pausas, "
            "las comas, y también las frases más largas sin puntuación clara"
        )
        palabras = texto_completo.split(" ")

        send_task = asyncio.create_task(_send_words(ws, palabras))

        # 3. Recibe eventos y audio hasta que el servidor confirme fin de turno.
        pcm_data = bytearray()
        t_start = time.monotonic()
        first_audio_received = False

        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                if not first_audio_received:
                    elapsed_ms = (time.monotonic() - t_start) * 1000
                    print(f"[latencia] primer audio recibido a los {elapsed_ms:.0f} ms")
                    first_audio_received = True
                pcm_data.extend(message)
            else:
                event = json.loads(message)
                print("[evento]", event)
                if event.get("event") == "turn_complete":
                    break

        await send_task
        _save_wav(out_path, bytes(pcm_data), sample_rate)
        print(f"Audio guardado en: {out_path}")


async def _send_words(ws, palabras: list[str]) -> None:
    """Envía las palabras una a una, simulando la cadencia de un LLM."""
    for palabra in palabras:
        await ws.send(json.dumps({"action": "text_chunk", "text": palabra + " "}))
        await asyncio.sleep(0.05)  # simula latencia entre tokens del LLM
    await ws.send(json.dumps({"action": "end_of_turn"}))


async def run_interrupt_demo(uri: str, voice_id: str) -> None:
    """Demuestra cómo interrumpir una síntesis en curso a mitad de camino."""
    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(
            json.dumps(
                {
                    "action": "configure",
                    "voice_params": {"voice_id": voice_id, "speed": 1.0},
                }
            )
        )
        print("[configure_ack]", json.loads(await ws.recv()))

        texto_largo = (
            "Esta es una frase deliberadamente larga para dar tiempo a que "
            "el cliente la interrumpa antes de que termine de sintetizarse por completo."
        )
        await ws.send(json.dumps({"action": "text_chunk", "text": texto_largo}))
        await ws.send(json.dumps({"action": "end_of_turn"}))

        chunks_recibidos = 0
        async for message in ws:
            if isinstance(message, bytes):
                chunks_recibidos += 1
                if chunks_recibidos == 3:
                    print("[cliente] enviando interrupt tras 3 chunks de audio...")
                    await ws.send(json.dumps({"action": "interrupt"}))
            else:
                event = json.loads(message)
                print("[evento]", event)
                if event.get("event") == "interrupted":
                    print(f"[cliente] confirmado: se recibieron {chunks_recibidos} "
                          f"chunks antes de la interrupción.")
                    break


def _save_wav(path: Path, pcm_data: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cliente de ejemplo para Voice Streaming API")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--voice", default="es_ES-davefx-medium")
    parser.add_argument("--out", default="output.wav")
    parser.add_argument(
        "--demo", choices=["basic", "interrupt"], default="basic",
        help="'basic' sintetiza un texto completo; 'interrupt' demuestra la cancelación",
    )
    args = parser.parse_args()

    uri = f"ws://{args.host}:{args.port}/ws/v1/audio-stream"

    if args.demo == "basic":
        asyncio.run(run_basic_demo(uri, args.voice, Path(args.out)))
    else:
        asyncio.run(run_interrupt_demo(uri, args.voice))


if __name__ == "__main__":
    main()
