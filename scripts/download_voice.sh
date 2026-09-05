#!/usr/bin/env bash
#
# download_voice.sh
# ------------------
# Descarga un modelo de voz de Piper TTS (archivo .onnx y .onnx.json) desde
# el repositorio oficial de Hugging Face y lo coloca en ./models/.
#
# Uso:
#   ./scripts/download_voice.sh es_ES-davefx-medium
#
# Puedes ver el catálogo completo de voces disponibles en:
#   https://github.com/rhasspy/piper/blob/master/VOICES.md
#   https://huggingface.co/rhasspy/piper-voices

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Uso: $0 <voice_id>   (ej: $0 es_ES-davefx-medium)"
  exit 1
fi

VOICE_ID="$1"

# El voice_id tiene el formato <idioma>_<PAIS>-<nombre>-<calidad>
# p. ej. es_ES-davefx-medium -> familia de idioma "es_ES", nombre "davefx",
# calidad "medium". La ruta en Hugging Face sigue esa misma jerarquía.
LANG_FAMILY=$(echo "$VOICE_ID" | cut -d'-' -f1 | cut -d'_' -f1)
LANG_COUNTRY=$(echo "$VOICE_ID" | cut -d'-' -f1)
VOICE_NAME=$(echo "$VOICE_ID" | cut -d'-' -f2)
QUALITY=$(echo "$VOICE_ID" | cut -d'-' -f3)

BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/${LANG_FAMILY}/${LANG_COUNTRY}/${VOICE_NAME}/${QUALITY}"

mkdir -p models
echo "Descargando ${VOICE_ID}.onnx ..."
curl -L -o "models/${VOICE_ID}.onnx" "${BASE_URL}/${VOICE_ID}.onnx"
echo "Descargando ${VOICE_ID}.onnx.json ..."
curl -L -o "models/${VOICE_ID}.onnx.json" "${BASE_URL}/${VOICE_ID}.onnx.json"

echo "Listo. Modelo guardado en models/${VOICE_ID}.onnx"
