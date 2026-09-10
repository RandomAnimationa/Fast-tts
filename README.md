# Fast-tts — Voice Streaming API



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
  (`length_scale`, `noise_scale`, `noise_w_scale`).
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
git clone https://github.com/RandomAnimationa/Fast-tts.git
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

Esto descarga `models/<voice_id>.onnx` y su archivo de configuración
`.onnx.json`. Puedes explorar el catálogo completo de voces en el
[repositorio de voces de Piper](https://huggingface.co/rhasspy/piper-voices),
en [VOICES.md](https://github.com/rhasspy/piper/blob/master/VOICES.md) y
probarlas antes de descargarlas en el
[Voice tester](https://rhasspy.github.io/piper-samples/).

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

---

## Protocolo WebSocket

Endpoint: **`ws://<host>:<port>/ws/v1/audio-stream`**

### 1. Configuración inicial (cliente → servidor)

Debe ser el primer mensaje tras abrir la conexión.

```json
{
  "action": "configure",
  "voice_params": {
    "voice_id": "es_ES-davefx-medium",
    "speed": 1.1,
    "pitch": 1.0,
    "emotion": "happy"
  }
}
```

| Parámetro | Rango | Descripción |
|---|---|---|
| `voice_id` | string | Nombre base del modelo (debe existir en `models/`) |
| `speed` | 0.5 – 2.0 | Velocidad del habla. Se mapea a `length_scale` (inverso) |
| `pitch` | 0.1 – 1.5 | Variabilidad tonal. Se mapea a `noise_scale` |
| `emotion` | `neutral`\|`happy`\|`sad`\|`serious` | Preset combinado de `length_scale`+`noise_scale`+`noise_w_scale` |

Respuesta del servidor:

```json
{ "event": "configure_ack", "sample_rate": 22050, "audio_format": "pcm_s16le_mono" }
```

### 2. Streaming de texto (cliente → servidor)

Se envía continuamente a medida que el LLM genera tokens:

```json
{ "action": "text_chunk", "text": "Hola, " }
```

El servidor fragmenta internamente el texto acumulado por frase (ver
`text_buffer.py`) y encola cada frase lista para síntesis en cuanto detecta
un separador adecuado — no hace falta que el cliente envíe frases completas.

Al terminar el turno (para forzar el flush de cualquier resto de texto que no
haya cerrado con puntuación):

```json
{ "action": "end_of_turn" }
```

El servidor responde `{"event": "turn_complete"}` cuando ha terminado de
encolar todo el texto pendiente (no implica que el audio ya se haya
reproducido por completo, solo que no queda texto sin encolar).

### 3. Recepción de audio (servidor → cliente)

El servidor alterna dos tipos de frames sobre el mismo socket:

- **Frames binarios**: bloques de audio **PCM crudo, 16-bit signed
  little-endian, mono**, a la tasa de muestreo indicada en `configure_ack`
  (normalmente 22050 Hz). Sin cabecera WAV — el cliente debe conocer el
  formato de antemano (por eso se informa en `configure_ack`).
- **Frames de texto (JSON)** con eventos de control:

```json
{ "event": "sentence_start", "id": 1 }
{ "event": "sentence_end", "id": 1 }
```

`id` corresponde al número secuencial de frase dentro de la sesión, útil
para que el cliente sincronice subtítulos o animaciones con el audio.

### 4. Interrupción / cancelación (cliente → servidor)

```json
{ "action": "interrupt" }
```

El servidor, al recibirlo:

1. Invalida la síntesis en curso mediante un flag de "sesión de habla"
   (`CancelFlag`), revisado entre bloques por el hilo de Piper — no espera a
   que termine de generar la frase actual.
2. Vacía la cola de texto pendiente (`text_queue`).
3. Descarta cualquier texto acumulado en el buffer de frase sin cortar aún.
4. Responde `{"event": "interrupted"}` para que el cliente sincronice su
   reproductor local (p. ej. resetear el *scheduler* de audio).

---

## Notas de diseño y limitaciones conocidas

- **API de `piper-tts`**: este proyecto usa `PiperVoice.synthesize()` (que
  entrega objetos `AudioChunk` con `.audio_int16_bytes`) y
  `piper.config.SynthesisConfig(length_scale, noise_scale, noise_w_scale)`,
  la API pública confirmada en las versiones recientes del paquete
  `piper-tts` (>=1.2). Si usas una versión muy distinta y ves un
  `AttributeError`, revisa `tts_engine.py::TTSEngine.synthesize_stream` y
  ajusta el nombre del método/objeto de retorno según `dir(PiperVoice)` en tu
  entorno.
- **Latencia real vs. objetivo de 150 ms**: con modelos Piper `medium` en 1
  vCPU, el tiempo de síntesis de la *primera* frase suele rondar 150-400 ms
  dependiendo de su longitud. Si tu caso de uso exige el límite inferior,
  considera usar un modelo `low`/`x_low` o bajar `min_first_chunk_len` en
  `SentenceBuffer` para emitir la primera frase más temprano.
- **Hilos vs. paralelismo interno de ONNX Runtime**: si tu modelo/entorno
  configura `intra_op_num_threads` en el runtime de ONNX, evita además lanzar
  demasiados workers en paralelo — puede generar sobre-suscripción de CPU en
  vez de más rendimiento. Ajusta `ThreadPoolExecutor`/concurrencia según tus
  núcleos reales disponibles.
- **`noise_w_scale`**: no todos los checkpoints de Piper exponen este
  parámetro de forma perceptible en tiempo de inferencia; verifica contra tu
  modelo específico si notas que cambiar la emoción no afecta demasiado el
  ritmo.
- **Un solo proceso, un solo worker**: pensado para instancias pequeñas. Para
  escalar horizontalmente, se recomienda levantar varias instancias detrás de
  un balanceador con *sticky sessions* (WebSockets), en vez de aumentar
  `--workers` en Uvicorn (duplicaría los modelos en RAM).

---

