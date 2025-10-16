import os
import json
import platform
import queue
import threading
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import soundcard as sc          # -> Salidas (WASAPI loopback)
import sounddevice as sd        # -> Entradas tipo "Mezcla estéreo" (fallback)

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from faster_whisper import WhisperModel

# ===================== Parámetros ajustables =====================
SAMPLE_RATE = 16000            # Whisper usa 16 kHz
CHUNK_SEC = 4.0                # 3–5 s recomendado (latencia vs. calidad)
OVERLAP_SEC = 0.8              # superposición entre chunks para continuidad
MODEL_SIZE = "small"           # "tiny", "base", "small", "medium", "large-v3"
LANG = "es"                    # Forzar español para reducir cambios de idioma y outros en inglés
VAD_FILTER = True              # Filtrar silencios (mejora subtítulos en vivo)
MAX_GUI_LINES = 200            # Mantener el widget de texto liviano
APP_TITLE = "Subtítulos en vivo (WASAPI loopback + faster-whisper)"
CONFIG_NAME = "subtitula_config.json"
SILENCE_RMS = 1e-4             # Umbral simple de silencio para evitar alucinaciones en pausas
# ================================================================

GOOD_INPUT_KEYWORDS = [
    "mezcla estéreo", "stereo mix", "loopback", "monitor"
]
ALLOWED_SD_HOSTS = {"Windows WASAPI", "Windows DirectSound", "MME"}

# ======================= Utilidades varias =======================

def get_config_path() -> str:
    base = os.path.join(os.path.expanduser("~"), ".config")
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        base = os.path.expanduser("~")
    return os.path.join(base, CONFIG_NAME)


def load_last_selection() -> Optional[str]:
    path = get_config_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("last_selection")
        except Exception:
            return None
    return None


def save_last_selection(sel: str) -> None:
    path = get_config_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"last_selection": sel}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Re-muestreo lineal con numpy (rápido y sin SciPy). x debe ser mono float32."""
    if src_sr == dst_sr or x.size == 0:
        return x.astype(np.float32, copy=False)
    duration = x.shape[0] / float(src_sr)
    n_dst = int(round(duration * dst_sr))
    if n_dst <= 1:
        return np.zeros((0,), dtype=np.float32)
    src_t = np.linspace(0.0, duration, num=x.shape[0], endpoint=False)
    dst_t = np.linspace(0.0, duration, num=n_dst, endpoint=False)
    y = np.interp(dst_t, src_t, x).astype(np.float32)
    return y


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.astype(np.float32, copy=False)
    return np.mean(x, axis=1).astype(np.float32)


# ==================== Listado de dispositivos ====================
@dataclass
class DeviceEntry:
    kind: str          # "spk" o "in"
    idx: int           # índice (para spk: enumeración propia; para in: índice sounddevice)
    name: str
    detail: str        # texto adicional

    def display(self) -> str:
        if self.kind == "spk":
            return f"spk:{self.idx}: {self.name} (WASAPI loopback)"
        else:
            return f"in:{self.idx}: {self.name}"


def list_devices() -> List[DeviceEntry]:
    devices: List[DeviceEntry] = []

    # 1) Salidas WASAPI (loopback) con soundcard, en orden preferente
    spk_list = list(sc.all_speakers())
    for i, spk in enumerate(spk_list):
        name = spk.name
        devices.append(DeviceEntry("spk", i, name, "WASAPI loopback"))

    # 2) Entradas "útiles" (Mezcla estéreo/monitor) con sounddevice
    try:
        sd_devs = sd.query_devices()
        hostapis = sd.query_hostapis()
        for i, d in enumerate(sd_devs):
            api = hostapis[d['hostapi']]['name']
            if api not in ALLOWED_SD_HOSTS:
                continue
            max_in = int(d.get('max_input_channels', 0) or 0)
            if max_in <= 0:
                continue
            lname = d['name'].lower()
            if any(k in lname for k in GOOD_INPUT_KEYWORDS):
                devices.append(DeviceEntry("in", i, d['name'], f"{api}"))
    except Exception:
        pass

    return devices

# =========================== App principal ===========================
class RealTimeTranscriber:
    def __init__(self):
        self.model = None

        # Estados
        self.running = False
        self.audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.capture_thread: Optional[threading.Thread] = None
        self.stream = None  # para sounddevice
        self.device_sr: int = 48000
        self.capture_kind: Optional[str] = None  # "spk" o "in"
        self.capture_info: Optional[str] = None

        # GUI
        self.root = None
        self.txt_widget: Optional[tk.Text] = None
        self.status_var = None
        self.start_btn = None
        self.stop_btn = None
        self.save_btn = None
        self.refresh_btn = None
        self.device_combo = None
        self.selected_device = None

        # Dispositivos y selección
        self.devices: List[DeviceEntry] = []
        self.selected_display: str = ""

        # Transcripción / timestamps
        self.transcript_plain: List[str] = []
        self.segments_abs: List[Tuple[float, float, str]] = []  # (t0, t1, text) en segundos
        self.total_sec_processed: float = 0.0  # para llevar offset absoluto por chunk
        self._final_buffer = np.zeros(0, dtype=np.float32)
        self._prev_context: str = ""  # contexto textual previo para initial_prompt


    def ensure_model_loaded(self):
        if self.model is None:
            self.update_status("Cargando modelo Whisper (la primera vez puede tardar)…")
            if self.root:
                self.root.update_idletasks()
            # Fuerza CPU e INT8 para portabilidad (sin DLLs de CUDA)
            self.model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
            self.update_status("Modelo listo.")


    # -------------------- Captura de audio --------------------
    def _producer_spk(self, spk_entry: DeviceEntry):
        """
        Captura el audio de salida (WASAPI loopback) del altavoz seleccionado.
        1) INTENTO A: sounddevice+WASAPI loopback
        2) INTENTO B (fallback): soundcard loopback (requiere numpy<2 si aparece
           el error de 'fromstring').
        """
        assert spk_entry.kind == "spk"

        # Emparejar speaker elegido
        speakers = list(sc.all_speakers())
        if not (0 <= spk_entry.idx < len(speakers)):
            self.update_status("Índice de altavoz inválido")
            return
        spk = speakers[spk_entry.idx]

        # ---------- INTENTO A: sounddevice (WASAPI loopback) ----------
        try:
            sd_devs = sd.query_devices()
            hostapis = sd.query_hostapis()
            target_idx = None
            for i, d in enumerate(sd_devs):
                api = hostapis[d['hostapi']]['name']
                if api != 'Windows WASAPI':
                    continue
                if int(d.get('max_output_channels', 0) or 0) <= 0:
                    continue
                if spk.name.lower() in d['name'].lower():
                    target_idx = i
                    break
            if target_idx is None:
                for i, d in enumerate(sd_devs):
                    api = hostapis[d['hostapi']]['name']
                    if api == 'Windows WASAPI' and int(d.get('max_output_channels', 0) or 0) > 0:
                        target_idx = i
                        break
            if target_idx is not None:
                dev_out = sd.query_devices(target_idx)
                self.device_sr = int(dev_out.get('default_samplerate') or 48000)

                # WasapiSettings según versión
                ws = None
                try:
                    ws = sd.WasapiSettings(loopback=True)
                    stream_kwargs = dict(extra_settings=ws)
                except TypeError:
                    try:
                        ws = sd.WasapiSettings(exclusive=False, loopback=True)
                        stream_kwargs = dict(extra_settings=ws)
                    except TypeError:
                        try:
                            ws = sd.WasapiSettings()
                            stream_kwargs = dict(extra_settings=ws)
                        except Exception:
                            ws = None
                            stream_kwargs = {}

                max_out_ch = int(dev_out.get('max_output_channels') or 0)
                channels_to_try = [c for c in [max_out_ch, 2, 1] if c and c > 0]
                tried = []
                for ch in channels_to_try:
                    try:
                        self.stream = sd.InputStream(
                            device=target_idx,
                            samplerate=self.device_sr,
                            channels=ch,
                            dtype='float32',
                            callback=self._sd_callback,
                            blocksize=0,
                            latency='low',
                            **stream_kwargs,
                        )
                        self.stream.start()
                        self.update_status(
                            f"Loopback WASAPI activo (sounddevice): '{dev_out['name']}' (ch={ch}, sr={self.device_sr})"
                        )
                        self.capture_info = f"Loopback WASAPI (sd): {dev_out['name']}"
                        return
                    except Exception as e:
                        tried.append((ch, str(e)))
                        self.stream = None
                self.update_status("Loopback WASAPI (sounddevice) no abrió, probando fallback. Intentos: " + str(tried))
        except Exception as e:
            self.update_status(f"Error usando sounddevice loopback, probando fallback: {e}")

        # ---------- INTENTO B (fallback): soundcard loopback ----------
        try:
            loop_mic = sc.get_microphone(id=spk.id, include_loopback=True)
            if loop_mic is None:
                self.update_status("No se pudo obtener loopback (soundcard).")
                return
            target_sr = 48000
            self.device_sr = target_sr
            block = int(target_sr * 0.1)
            self.update_status(f"Loopback WASAPI activo (soundcard): '{spk.name}' (sr={target_sr})")
            self.capture_info = f"Loopback WASAPI (soundcard): {spk.name}"
            with loop_mic.recorder(samplerate=target_sr, channels=2, blocksize=block) as rec:
                while self.running:
                    try:
                        data = rec.record(numframes=block)
                        mono = to_mono(data)
                        self.audio_q.put(mono)
                    except Exception as e:
                        msg = str(e)
                        if 'fromstring' in msg and 'frombuffer' in msg:
                            self.update_status(
                                "Error de loopback (soundcard) por NumPy 2.x. Ejecuta: pip install \"numpy<2\""
                            )
                        else:
                            self.update_status(f"Error de captura (soundcard loopback): {e}")
                        break
        except Exception as e:
            self.update_status(f"No se pudo abrir loopback con soundcard: {e}")

    def _sd_callback(self, indata, frames, time_info, status):
        if status:
            pass
        mono = to_mono(indata)
        self.audio_q.put(mono)

    def _producer_in(self, in_entry: DeviceEntry):
        """Abre una entrada (Mezcla estéreo / monitor) con sounddevice."""
        assert in_entry.kind == "in"
        try:
            devinfo = sd.query_devices(in_entry.idx)
            hostapis = sd.query_hostapis()
            api = hostapis[devinfo['hostapi']]['name']
            self.device_sr = int(devinfo.get('default_samplerate') or 48000)

            tried = []
            for ch in ([2, 1] if int(devinfo.get('max_input_channels', 0) or 0) >= 2 else [1]):
                try:
                    self.stream = sd.InputStream(
                        device=in_entry.idx,
                        samplerate=self.device_sr,
                        channels=ch,
                        dtype="float32",
                        callback=self._sd_callback,
                        blocksize=0,
                        latency="low",
                    )
                    self.stream.start()
                    self.update_status(f"Capturando entrada: '{devinfo['name']}' (canales={ch}, sr={self.device_sr}, {api})")
                    self.capture_info = f"Entrada: {devinfo['name']} ({api})"
                    return
                except Exception as e:
                    tried.append((ch, str(e)))
                    self.stream = None
            raise RuntimeError("No se pudo abrir el dispositivo de entrada. Intentos: " + str(tried))
        except Exception as e:
            messagebox.showerror("Error de audio", str(e))
            self.update_status(f"Error entrada: {e}")
            self.running = False

    # ----------------- Consumo + transcripción -----------------
    def consumer_loop(self):
        buffer = np.zeros(0, dtype=np.float32)
        last_emit = time.time()
        src_sr = self.device_sr
        overlap_samples_src = max(0, int(OVERLAP_SEC * src_sr))
        while self.running:
            try:
                data = self.audio_q.get(timeout=0.1)
                buffer = np.concatenate([buffer, data])
            except queue.Empty:
                pass

            # Emitimos por CHUNK_SEC (según reloj de pared)
            if time.time() - last_emit >= CHUNK_SEC and buffer.size > 0:
                chunk_src = buffer.copy()
                # Prepara el búfer para el próximo ciclo conservando la superposición
                buffer = buffer[-overlap_samples_src:] if overlap_samples_src > 0 else np.zeros(0, dtype=np.float32)
                last_emit = time.time()

                # Re-muestreo a 16 kHz + envío a Whisper
                mono16 = resample_linear(chunk_src, src_sr, SAMPLE_RATE)
                if mono16.size == 0:
                    continue

                # Filtro de silencio sobre la parte NO superpuesta
                if overlap_samples_src > 0:
                    main_part = chunk_src[:-overlap_samples_src] if chunk_src.size > overlap_samples_src else chunk_src
                else:
                    main_part = chunk_src
                main16 = resample_linear(main_part, src_sr, SAMPLE_RATE)
                rms = float(np.sqrt(np.mean(main16**2))) if main16.size else 0.0
                if rms < SILENCE_RMS:
                    # Avanza el tiempo procesado sin transcribir (silencio real)
                    self.total_sec_processed += max(0, (len(chunk_src) - overlap_samples_src)) / float(src_sr)
                    continue

                self.update_status("Transcribiendo…")
                try:
                    instruction = (
                        "El audio está en español latino. Transcribe EXACTAMENTE lo que se oye, "
                        "sin agregar saludos, cierres, URLs ni frases como 'gracias por ver'."
                    )
                    context_tail = self._prev_context[-400:]
                    initial_prompt = instruction if not context_tail else f"{instruction}\nContexto previo: {context_tail}"

                    segments, _ = self.model.transcribe(
                        mono16,
                        language=LANG,
                        vad_filter=True,
                        beam_size=5,
                        temperature=0.0,
                        condition_on_previous_text=True,
                        compression_ratio_threshold=2.4,
                        no_speech_threshold=0.45,
                        log_prob_threshold=-0.5,
                        initial_prompt=initial_prompt,
                    )
                    piece_list = []
                    for seg in segments:
                        t0 = float(seg.start) if seg.start is not None else 0.0
                        t1 = float(seg.end) if seg.end is not None else t0
                        abs_t0 = self.total_sec_processed + t0
                        abs_t1 = self.total_sec_processed + t1
                        txt = (seg.text or "").strip()
                        if txt:
                            self.segments_abs.append((abs_t0, abs_t1, txt))
                            piece_list.append(txt)
                    piece = " ".join(piece_list).strip()
                    if piece:
                        self.transcript_plain.append(piece)
                        # Actualiza contexto textual
                        self._prev_context = (self._prev_context + " " + piece).strip()
                        self._gui_append(piece)
                        self.update_status("Listo.")
                except Exception as e:
                    self.update_status(f"Error al transcribir: {e}")

                # Avanza el offset absoluto por la duración del chunk menos la superposición
                self.total_sec_processed += max(0, (len(chunk_src) - overlap_samples_src)) / float(src_sr)

        # guardar residuo para finalizar al detener
        self._final_buffer = buffer

    # -------------------------- GUI --------------------------
    def _gui_append(self, piece: str):
        if self.txt_widget:
            self.txt_widget.configure(state="normal")
            self.txt_widget.insert(tk.END, piece + "\n")
            # Mantener ligero
            if int(self.txt_widget.index('end-1c').split('.')[0]) > MAX_GUI_LINES:
                self.txt_widget.delete("1.0", "2.0")
            self.txt_widget.see(tk.END)
            self.txt_widget.configure(state="disabled")

    def update_status(self, msg: str):
        if self.status_var is not None:
            self.status_var.set(msg)

    # ------------------ Finalizar procesamiento y guardado ------------------
    def _finalize_processing(self):
        # Recoge cualquier residuo del búfer + cola y lo procesa con el mismo esquema de solape
        src_sr = self.device_sr
        overlap_samples_src = max(0, int(OVERLAP_SEC * src_sr))
        buffer = getattr(self, "_final_buffer", np.zeros(0, dtype=np.float32))
        while True:
            try:
                data = self.audio_q.get_nowait()
                buffer = np.concatenate([buffer, data])
            except queue.Empty:
                break
        if buffer.size == 0:
            self.update_status("Listo.")
            return

        # Procesa en bloques con el mismo tamaño que en vivo
        block_len = int(CHUNK_SEC * src_sr)
        pos = 0
        while pos < len(buffer):
            end = min(len(buffer), pos + block_len)
            chunk_src = buffer[pos:end]
            # siguiente posición con solape
            pos = end - overlap_samples_src if end - overlap_samples_src > pos else end

            mono16 = resample_linear(chunk_src, src_sr, SAMPLE_RATE)
            if mono16.size == 0:
                continue

            if overlap_samples_src > 0:
                main_part = chunk_src[:-overlap_samples_src] if chunk_src.size > overlap_samples_src else chunk_src
            else:
                main_part = chunk_src
            main16 = resample_linear(main_part, src_sr, SAMPLE_RATE)
            rms = float(np.sqrt(np.mean(main16**2))) if main16.size else 0.0
            if rms < SILENCE_RMS:
                self.total_sec_processed += max(0, (len(chunk_src) - overlap_samples_src)) / float(src_sr)
                continue

            try:
                self.update_status("Transcribiendo residuo…")
                instruction = (
                    "El audio está en español latino. Transcribe EXACTAMENTE lo que se oye, "
                    "sin agregar saludos, cierres, URLs ni frases como 'gracias por ver'."
                )
                context_tail = self._prev_context[-400:]
                initial_prompt = instruction if not context_tail else f"{instruction}\nContexto previo: {context_tail}"

                segments, _ = self.model.transcribe(
                    mono16,
                    language=LANG,
                    vad_filter=True,
                    beam_size=5,
                    temperature=0.0,
                    condition_on_previous_text=True,
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=0.45,
                    log_prob_threshold=-0.5,
                    initial_prompt=initial_prompt,
                )
                piece_list = []
                for seg in segments:
                    t0 = float(seg.start) if seg.start is not None else 0.0
                    t1 = float(seg.end) if seg.end is not None else t0
                    abs_t0 = self.total_sec_processed + t0
                    abs_t1 = self.total_sec_processed + t1
                    txt = (seg.text or "").strip()
                    if txt:
                        self.segments_abs.append((abs_t0, abs_t1, txt))
                        piece_list.append(txt)
                piece = " ".join(piece_list).strip()
                if piece:
                    self.transcript_plain.append(piece)
                    self._prev_context = (self._prev_context + " " + piece).strip()
                    self._gui_append(piece)
            except Exception:
                pass
            finally:
                self.total_sec_processed += max(0, (len(chunk_src) - overlap_samples_src)) / float(src_sr)
        self.update_status("Listo.")

    # ---------------------- Guardado de archivos ----------------------
    def save_transcript(self):
        all_text = "\n".join(self.transcript_plain).strip()
        default_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_name = f"transcript_{default_stamp}.txt"
        if not all_text:
            messagebox.showinfo("Vacío", "No hay texto para guardar.")
            return
        path_txt = filedialog.asksaveasfilename(
            title="Guardar transcript (.txt)",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("Texto", "*.txt")],
        )
        if path_txt:
            try:
                with open(path_txt, "w", encoding="utf-8") as f:
                    f.write(all_text + "\n")
                self.update_status(f"Guardado TXT: {path_txt}")
            except Exception as e:
                messagebox.showerror("Error al guardar TXT", str(e))

    def clear_screen(self):
        """Limpia la pantalla y borra el texto transcrito hasta ahora."""
        if self.txt_widget:
            self.txt_widget.configure(state="normal")
            self.txt_widget.delete("1.0", tk.END)
            self.txt_widget.configure(state="disabled")

        # También vaciamos los datos en memoria para evitar confusión
        self.transcript_plain.clear()
        self.segments_abs.clear()
        self._prev_context = ""
        self.total_sec_processed = 0.0

        self.update_status("Pantalla limpiada.")

    # ---------------------- Ciclo de vida ----------------------
    def refresh_devices(self, preselect_last: bool = True):
        self.devices = list_devices()
        values = [d.display() for d in self.devices]
        self.device_combo["values"] = values
        preferred = None

        if preselect_last:
            last = load_last_selection()
            if last and last in values:
                preferred = last

        if preferred is None:
            for s in values:
                if s.lower().startswith("spk:") and "logitech" in s.lower():
                    preferred = s
                    break
        if preferred is None:
            for s in values:
                if s.lower().startswith("spk:"):
                    preferred = s
                    break
        if preferred is None and values:
            preferred = values[0]

        if preferred:
            self.selected_device.set(preferred)
        self.update_status("Dispositivos actualizados.")

    def start(self):
        if self.running:
            return
        if platform.system() != "Windows":
            messagebox.showerror("No soportado", "Este modo está pensado para Windows (WASAPI).")
            return

        try:
            # ⬇️ ahora dentro del try
            self.ensure_model_loaded()

            sel = self.selected_device.get().strip()
            if not sel:
                messagebox.showwarning("Selecciona dispositivo", "No hay dispositivo seleccionado.")
                return

            save_last_selection(sel)
            self.running = True
            self.total_sec_processed = 0.0
            self.transcript_plain.clear()
            self.segments_abs.clear()
            self._final_buffer = np.zeros(0, dtype=np.float32)
            self._prev_context = ""

            if sel.lower().startswith("spk:"):
                self.capture_kind = "spk"
                spk_idx = int(sel.split(":", 2)[1])
                entry = DeviceEntry("spk", spk_idx, sel, "WASAPI loopback")
                self.capture_thread = threading.Thread(target=self._producer_spk, args=(entry,), daemon=True)
                self.capture_thread.start()
            elif sel.lower().startswith("in:"):
                self.capture_kind = "in"
                in_idx = int(sel.split(":", 2)[1])
                entry = DeviceEntry("in", in_idx, sel, "entrada")
                self._producer_in(entry)
            else:
                raise RuntimeError("Selecciona un dispositivo válido (spk:/in:)")

            threading.Thread(target=self.consumer_loop, daemon=True).start()
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.save_btn.configure(state="disabled")

        except Exception as e:
            self.running = False
            messagebox.showerror("Error de inicio", str(e))
            self.update_status(f"Error: {e}")


    def stop(self):
        if not self.running:
            return
        self.running = False
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        self.stream = None

        # Drenar cola y procesar residuo antes de habilitar guardar
        self.update_status("Deteniendo… procesando cola…")
        self._finalize_processing()

        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        # Habilitar guardar solo cuando ya está "Listo."
        self.save_btn.configure(state="normal")

    # ---------------------- Construcción GUI ----------------------
    def build_gui(self):
        self.root = tk.Tk()
        self.root.title(APP_TITLE)

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        # Selección de dispositivo
        ttk.Label(main, text="Dispositivo (prioriza 'spk:' para WASAPI loopback del Logitech G733):").pack(anchor="w")

        self.selected_device = tk.StringVar(value="")
        self.device_combo = ttk.Combobox(main, textvariable=self.selected_device, values=[], state="readonly")
        self.device_combo.pack(fill="x", pady=(0, 6))

        # Botonera de dispositivos
        dbar = ttk.Frame(main)
        dbar.pack(fill="x", pady=(0, 10))
        self.refresh_btn = ttk.Button(dbar, text="Actualizar dispositivos", command=lambda: self.refresh_devices(preselect_last=True))
        self.refresh_btn.pack(side="left")

        # Cargar lista al inicio
        self.refresh_devices(preselect_last=True)

        # Texto de subtítulos
        self.txt_widget = tk.Text(main, height=18, wrap="word", state="disabled")
        self.txt_widget.pack(fill="both", expand=True)

        # Botonera principal
        btns = ttk.Frame(main)
        btns.pack(fill="x", pady=(8, 8))
        self.start_btn = ttk.Button(btns, text="Iniciar", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="Detener", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.clear_btn = ttk.Button(btns, text="Limpiar pantalla", command=self.clear_screen)
        self.clear_btn.pack(side="left", padx=6)
        self.save_btn = ttk.Button(btns, text="Guardar (.txt)", command=self.save_transcript, state="disabled")
        self.save_btn.pack(side="left", padx=6)

        # Barra de estado
        self.status_var = tk.StringVar(value="Listo.")
        ttk.Label(main, textvariable=self.status_var).pack(anchor="w")

        # Cierre seguro
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def on_close(self):
        if self.running:
            if not messagebox.askyesno("Salir", "Está grabando. ¿Detener y salir?"):
                return
            self.stop()
        self.root.destroy()


if __name__ == "__main__":
    app = RealTimeTranscriber()
    app.build_gui()
