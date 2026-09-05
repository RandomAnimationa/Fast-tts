# Voice Streaming API

API de **streaming de voz (Text-to-Speech)** de baja latencia sobre **FastAPI +
WebSockets**, diseñada para ejecutarse en servidores con **CPU únicamente**
(sin GPU dedicada), usando **Piper TTS** (VITS exportado a ONNX) como motor de
síntesis.

Ideal para conectarse a la salida en streaming de un LLM: el texto entra
token a token y el audio empieza a reproducirse antes de que el LLM termine
de generar la respuesta completa.

---

## Características

- 🔊 **Streaming bidireccional**: entra texto progresivo, sale audio progresivo.
- ⚡ **Baja latencia**: fragmentación de texto por frase + síntesis en hilos
  (`asyncio.to_thread`/`ThreadPoolExecutor`) sin bloquear el event loop.
- 🧠 **Sin GPU**: pensado para correr con 1 vCPU y ~150 MB de RAM base.
- 🎛️ **Control de prosodia**: `speed`, `pitch` y `emotion` (presets `neutral`,
  `happy`, `sad`, `serious`) mapeados a los parámetros nativos de VITS
  (`length_scale`, `noise_scale`, `noise_w`).
- ⏹️ **Interrupciones**: cancelación cooperativa a mitad de síntesis mediante
  un flag de "sesión de habla", sin dejar hilos huérfanos.
- 📦 **Cero dependencias de disco en el camino caliente**: todo el pipeline
  vive en colas `asyncio.Queue` en memoria.
- 🧪 **Cliente de ejemplo incluido**: CLI en Python que ejercita todo el
  protocolo (configuración, envío progresivo de texto, recepción de audio,
  interrupción) y guarda el resultado en un `.wav`.

---

## Estructura del proyecto

```
Fast-tts/
├── main.py                    # App FastAPI, WebSocket, orquestación de la sesión
├── tts_engine.py               # Wrapper de Piper TTS (carga de modelos, síntesis por streaming)
├── text_buffer.py               # Fragmentación de texto por puntuación (chunking)
├── requirements.txt
├── scripts/
│   └── download_voice.sh        # Descarga modelos .onnx de voces Piper
├── models/                      # Modelos .onnx / .onnx.json (no versionados en git)
└── client_example/
    └── client.py                 # Cliente CLI en Python (websockets)
```

---

## Instalación

```bash
git clone <este-repo>
cd Fast-tts

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Descargar un modelo de voz

Mujer
```bash
./scripts/download_voice.sh es_MX-claude-high
```

Hombre
```bash
./scripts/download_voice.sh es_ES-davefx-medium
```

Esto descarga `models/es_ES-davefx-medium.onnx` y su archivo de
configuración `.onnx.json`. Puedes explorar el catálogo completo de voces en
el [repositorio de voces de Piper](https://huggingface.co/rhasspy/piper-voices), 
[VOICES.md](https://github.com/rhasspy/piper/blob/master/VOICES.md) y en 
[Voice tester](https://rhasspy.github.io/piper-samples/)



Si quieres precargar varias voces al arrancar, edita la lista
`PRELOAD_VOICES` en `main.py`.

---

## Ejecutar el servidor

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

> **¿Por qué `--workers 1`?** La concurrencia entre sesiones se maneja
> internamente con `asyncio` + hilos de síntesis, no con múltiples procesos.
> Usar 1 solo worker minimiza el consumo de RAM (cada worker de Uvicorn
> cargaría su propia copia de los modelos en memoria).

Verifica que está vivo:

```bash
curl http://localhost:8000/health
```

Documentación automática de los endpoints HTTP auxiliares (no del protocolo
WebSocket, que se describe abajo): `http://localhost:8000/docs`.

---

## Probar con los clientes de ejemplo

### Cliente Python (CLI)

```bash
cd client_example
pip install websockets
python client.py --host localhost --port 8000 --voice es_ES-davefx-medium --out salida.wav
```

Esto sintetiza un texto de prueba enviado palabra por palabra (simulando un
LLM), mide e imprime la **latencia hasta el primer audio**, y guarda el
resultado en `salida.wav`.

Para probar la cancelación a mitad de síntesis:

```bash
python client.py --demo interrupt
```

