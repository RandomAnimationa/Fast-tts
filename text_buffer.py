"""
text_buffer.py
==============

Algoritmo de fragmentación de texto (chunking) para síntesis de voz en streaming.

El objetivo es convertir un flujo continuo de tokens/palabras (tal como los
genera un LLM) en fragmentos de texto "pronunciables" lo antes posible, sin
esperar a que la respuesta completa esté disponible.

Reglas de corte (en orden de prioridad):
    1. Separadores primarios: '.', '?', '!', '\n'  -> corte inmediato.
    2. Separadores secundarios: ',', ';', ':'       -> corte si el acumulador
       ya tiene más de `secondary_cut_len` caracteres.
    3. Longitud máxima (fallback de latencia): si el acumulador supera
       `max_chunk_len` caracteres sin encontrar puntuación, se corta en el
       último espacio disponible (para no partir una palabra a la mitad).

Uso típico:

    buffer = SentenceBuffer()
    for token in llm_stream:
        for sentence in buffer.feed(token):
            await text_queue.put(sentence)
    # Al finalizar el stream del LLM, se debe vaciar lo que quede:
    for sentence in buffer.flush():
        await text_queue.put(sentence)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Separadores que provocan un corte inmediato de la frase.
_PRIMARY_SEPARATORS = (".", "?", "!", "\n")

# Separadores que provocan un corte solo si el acumulador ya es "suficientemente
# largo" (para evitar cortar frases como "Sí, claro" en fragmentos absurdos).
_SECONDARY_SEPARATORS = (",", ";", ":")

# Se usa para encontrar el último espacio "seguro" donde cortar por longitud.
_WHITESPACE_RE = re.compile(r"\s")


@dataclass
class SentenceBuffer:
    """Acumula texto entrante y emite fragmentos listos para sintetizar.

    Atributos configurables:
        secondary_cut_len: longitud mínima del acumulador para que un
            separador secundario (,;:) dispare un corte. Por defecto 25.
        max_chunk_len: longitud máxima del acumulador antes de forzar un
            corte por espacio (fallback de latencia). Por defecto 60.
        min_first_chunk_len: longitud mínima deseada para el *primer*
            fragmento de una respuesta nueva. Al bajar este valor se reduce
            la latencia hasta el primer audio, a costa de fragmentos más
            cortos y potencialmente menos naturales. Por defecto 20.
    """

    secondary_cut_len: int = 25
    max_chunk_len: int = 60
    min_first_chunk_len: int = 20

    _acc: str = field(default="", init=False, repr=False)
    _emitted_any: bool = field(default=False, init=False, repr=False)

    def reset(self) -> None:
        """Limpia el acumulador (usar tras una interrupción/cancelación)."""
        self._acc = ""
        self._emitted_any = False

    def feed(self, token: str) -> list[str]:
        """Alimenta el buffer con un nuevo token/palabra/chunk de texto.

        Devuelve una lista (posiblemente vacía) de fragmentos listos para
        enviar a la cola de síntesis. Puede devolver más de un fragmento si
        el token recibido contiene varios separadores.
        """
        self._acc += token
        return self._drain()

    def flush(self) -> list[str]:
        """Fuerza la emisión de cualquier texto restante en el acumulador.

        Debe llamarse al finalizar el stream de texto del LLM (fin de turno)
        para no perder la última frase, que puede no terminar en puntuación.
        """
        out = []
        remainder = self._acc.strip()
        if remainder:
            out.append(remainder)
        self._acc = ""
        self._emitted_any = False
        return out

    # -- lógica interna -----------------------------------------------------

    def _drain(self) -> list[str]:
        out: list[str] = []

        while True:
            cut_index = self._find_primary_cut()
            if cut_index is None:
                cut_index = self._find_secondary_cut()
            if cut_index is None:
                cut_index = self._find_length_fallback_cut()
            if cut_index is None:
                break

            sentence = self._acc[:cut_index].strip()
            self._acc = self._acc[cut_index:].lstrip()

            if sentence:
                out.append(sentence)
                self._emitted_any = True

        return out

    def _find_primary_cut(self) -> int | None:
        best = None
        for sep in _PRIMARY_SEPARATORS:
            idx = self._acc.find(sep)
            if idx != -1:
                candidate = idx + 1
                if best is None or candidate < best:
                    best = candidate
        return best

    def _find_secondary_cut(self) -> int | None:
        # Umbral más bajo para el primer fragmento de la respuesta: prioriza
        # la latencia percibida sobre la naturalidad de un fragmento largo.
        threshold = (
            self.min_first_chunk_len if not self._emitted_any else self.secondary_cut_len
        )
        if len(self._acc) < threshold:
            return None

        best = None
        for sep in _SECONDARY_SEPARATORS:
            idx = self._acc.find(sep)
            if idx != -1:
                candidate = idx + 1
                if best is None or candidate < best:
                    best = candidate
        return best

    def _find_length_fallback_cut(self) -> int | None:
        if len(self._acc) <= self.max_chunk_len:
            return None

        # Busca el último espacio dentro del límite para no partir palabras.
        window = self._acc[: self.max_chunk_len]
        match = None
        for m in _WHITESPACE_RE.finditer(window):
            match = m
        if match is not None:
            return match.end()

        # No hay espacios (palabra muy larga / texto pegado): corta a la fuerza.
        return self.max_chunk_len

    def __len__(self) -> int:
        return len(self._acc)

    @property
    def pending(self) -> str:
        """Texto actualmente acumulado sin emitir (solo lectura/depuración)."""
        return self._acc
