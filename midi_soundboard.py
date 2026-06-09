#!/usr/bin/env python3
"""
MIDI Piano & Soundboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Turns an Akai MPK Mini MKII into a real-time piano synthesizer + soundboard.

  • MIDI Ch 1  → additive-synthesis piano (velocity-sensitive, polyphonic)
  • MIDI Ch 10 → per-pad audio file playback (MP3 / WAV / OGG / FLAC)
  • Drag audio files onto pads — mappings auto-saved to config.json
  • Selectable MIDI input and audio output device (VB-Audio Cable support)
  • Continues working while minimised (MIDI runs in a background thread)

Dependencies:
    pip install PyQt6 mido python-rtmidi numpy sounddevice soundfile
"""

import sys
import os
import json
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import mido

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont


# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE   = 44100   # Hz
BLOCK_SIZE    = 256     # frames per audio callback (~5.8 ms latency)
MAX_POLYPHONY = 32      # oldest voice stolen when this limit is hit

# Akai MPK Mini MkII Bank-A pad → MIDI note mapping.
# The device has 8 pads in a 2×4 grid; top row is pads 5-8, bottom is 1-4.
PAD_NOTES   = [40, 41, 42, 43,   # row 0 (top)    – pads 5-8
               36, 37, 38, 39]   # row 1 (bottom)  – pads 1-4
NOTE_TO_PAD = {note: idx for idx, note in enumerate(PAD_NOTES)}
PAD_COLS    = 4   # pads per row

# Config is stored in the user's home directory so it survives app moves.
CONFIG_FILE = Path.home() / ".midi_soundboard" / "config.json"


# ── Config persistence ────────────────────────────────────────────────────────

class Config:
    """Loads and saves pad mappings + device selections to a JSON file."""

    def __init__(self):
        self.pad_files: dict[int, str] = {}   # pad_index (0-15) → file path
        self.midi_device: str  = ""
        self.audio_device: str = ""
        self._load()

    def _load(self):
        if not CONFIG_FILE.is_file():
            return
        try:
            raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            self.pad_files    = {int(k): v for k, v in raw.get("pads", {}).items()}
            self.midi_device  = raw.get("midi_device", "")
            self.audio_device = raw.get("audio_device", "")
        except Exception:
            pass  # corrupt config → start fresh

    def save(self):
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps({
            "pads":         {str(k): v for k, v in self.pad_files.items()},
            "midi_device":  self.midi_device,
            "audio_device": self.audio_device,
        }, indent=2), encoding="utf-8")


# ── Piano synthesis ───────────────────────────────────────────────────────────

def midi_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def note_name(note: int) -> str:
    names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    return f"{names[note % 12]}{note // 12 - 1}"


def synthesise_note(freq: float) -> np.ndarray:
    """
    Additive synthesis of a piano-like tone.
    Returns a stereo float32 array of shape (N, 2), peak ≤ 0.70.

    Technique: sum of harmonics, each with its own exponential decay rate
    (upper harmonics decay faster, giving a piano's characteristic brightness
    at the attack that fades to a pure fundamental).
    """
    # Lower notes ring longer than upper notes.
    duration  = float(np.clip(55_000 / (freq * 30), 0.8, 3.2))
    n_samples = int(duration * SAMPLE_RATE)
    t         = np.linspace(0.0, duration, n_samples, dtype=np.float32)

    # (harmonic multiple, relative amplitude, decay constant)
    partials = [(1, 1.00, 2.5), (2, 0.55, 3.5), (3, 0.30, 5.5),
                (4, 0.15, 8.0), (5, 0.07, 11.0), (6, 0.03, 15.0)]

    mono = np.zeros(n_samples, dtype=np.float32)
    for h, amp, decay in partials:
        if freq * h < SAMPLE_RATE / 2:          # Nyquist guard
            env   = np.exp(-decay * t / duration).astype(np.float32)
            mono += amp * env * np.sin(2 * np.pi * freq * h * t).astype(np.float32)

    # 3 ms linear ramp to eliminate the click at note onset.
    ramp = max(1, int(0.003 * SAMPLE_RATE))
    mono[:ramp] *= np.linspace(0.0, 1.0, ramp, dtype=np.float32)

    peak = float(np.max(np.abs(mono)))
    if peak:
        mono *= 0.70 / peak

    return np.column_stack([mono, mono])   # mono → (N, 2) stereo


class NoteCache:
    """
    Pre-generates all 88 piano-range notes (MIDI 21-108) in a background
    thread so they are ready for zero-latency lookup during play.
    """

    def __init__(self):
        self._data: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()

    def warm_up(self):
        """Call from a worker thread — blocks until all notes are ready."""
        for note in range(21, 109):
            self._get_unlocked(note)

    def get(self, note: int) -> np.ndarray:
        with self._lock:
            return self._get_unlocked(note)

    def _get_unlocked(self, note: int) -> np.ndarray:
        if note not in self._data:
            self._data[note] = synthesise_note(midi_to_hz(note))
        return self._data[note]


# ── Audio engine ──────────────────────────────────────────────────────────────

class Voice:
    """
    A cursor over a pre-computed stereo buffer.
    Used for both piano notes and soundboard samples.
    """
    __slots__ = ("buf", "pos", "gain", "done")

    def __init__(self, buf: np.ndarray, gain: float = 1.0):
        self.buf  = buf      # (N, 2) float32
        self.pos  = 0
        self.gain = np.float32(gain)
        self.done = False

    def read(self, n: int) -> np.ndarray:
        if self.done:
            return np.zeros((n, 2), dtype=np.float32)
        remaining = len(self.buf) - self.pos
        chunk     = self.buf[self.pos : self.pos + n]
        self.pos += n
        if self.pos >= len(self.buf):
            self.done = True
        # Zero-pad the final fragment so the caller always gets exactly n frames.
        if len(chunk) < n:
            chunk = np.pad(chunk, ((0, n - len(chunk)), (0, 0)))
        return chunk * self.gain


class AudioEngine:
    """
    Owns a low-latency PortAudio output stream via sounddevice.
    All public methods are thread-safe; they are called from the MIDI thread.
    """

    def __init__(self):
        self._voices : list[Voice]           = []
        self._samples: dict[str, np.ndarray] = {}   # path → (N,2) float32 cache
        self._lock   = threading.Lock()
        self._stream : sd.OutputStream | None = None

    # ── Life-cycle ─────────────────────────────────────────────────────────

    def start(self, device_index: int | None = None):
        self._stream = sd.OutputStream(
            device     = device_index,
            samplerate = SAMPLE_RATE,
            blocksize  = BLOCK_SIZE,
            channels   = 2,
            dtype      = "float32",
            latency    = "low",
            callback   = self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def restart(self, device_index: int | None = None):
        """Hot-swap the output device without losing the sample cache."""
        with self._lock:
            self._voices.clear()
        self.stop()
        self.start(device_index)

    # ── PortAudio callback (runs on the real-time audio thread) ────────────

    def _callback(self, out: np.ndarray, frames: int, _time, _status):
        mix = np.zeros((frames, 2), dtype=np.float32)
        with self._lock:
            for v in self._voices:
                mix += v.read(frames)
            self._voices = [v for v in self._voices if not v.done]
        # Hard-clip to [-1, 1] prevents distortion when many voices overlap.
        np.clip(mix, -1.0, 1.0, out=out)

    # ── Public play API ────────────────────────────────────────────────────

    def play_note(self, buf: np.ndarray, velocity: float):
        """Inject a piano voice. velocity is 0.0–1.0."""
        voice = Voice(buf, velocity * 0.8)   # 0.8 leaves headroom for polyphony
        with self._lock:
            if len(self._voices) >= MAX_POLYPHONY:
                self._voices.pop(0)           # steal the oldest voice
            self._voices.append(voice)

    def play_pad(self, path: str):
        """Trigger a soundboard sample at full volume."""
        buf = self._load(path)
        if buf is not None:
            with self._lock:
                self._voices.append(Voice(buf, 1.0))

    def preload(self, path: str):
        """Load a file into the cache so the first trigger has no latency."""
        self._load(path)

    # ── Sample loader ──────────────────────────────────────────────────────

    def _load(self, path: str) -> np.ndarray | None:
        """
        Load any audio file soundfile supports (WAV, FLAC, OGG, MP3*) into a
        resampled, stereo float32 array.  Result is cached in memory.

        *MP3 requires soundfile ≥ 0.12.1 (bundles libsndfile 1.1.0+).
        """
        if path in self._samples:
            return self._samples[path]
        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception as e:
            print(f"[audio] cannot load {path!r}: {e}")
            return None

        # Resample to the engine rate using linear interpolation.
        if sr != SAMPLE_RATE:
            n_new   = int(len(data) * SAMPLE_RATE / sr)
            old_idx = np.arange(len(data), dtype=np.float64)
            new_idx = np.linspace(0.0, len(data) - 1, n_new)
            data    = np.column_stack([
                np.interp(new_idx, old_idx, data[:, ch]).astype(np.float32)
                for ch in range(data.shape[1])
            ])

        # Normalise channel count to exactly 2.
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        elif data.shape[1] > 2:
            data = data[:, :2]

        self._samples[path] = data
        return data


# ── MIDI listener ─────────────────────────────────────────────────────────────

class MidiListener(QThread):
    """
    Background thread that keeps a MIDI port open and forwards messages as
    Qt signals.  Using Qt signals means all handling runs on the GUI thread,
    where it is safe to update widgets.

    mido's callback API is used so this thread only keeps the port alive;
    the OS/rtmidi delivers messages on their own internal thread.
    """

    note_on  = pyqtSignal(int, int, int)   # channel (1-based), note, velocity
    note_off = pyqtSignal(int, int)        # channel, note

    def __init__(self):
        super().__init__()
        self._device = ""
        self._alive  = True

    def set_device(self, name: str):
        self._device = name
        # The run() loop will pick up the new name on the next reconnect cycle.

    def run(self):
        while self._alive:
            port_name = self._choose_port()
            if not port_name:
                time.sleep(1.0)
                continue
            try:
                with mido.open_input(port_name, callback=self._on_msg) as _port:
                    # Block until the device disappears from the system.
                    while self._alive and port_name in mido.get_input_names():
                        time.sleep(0.25)
            except Exception as e:
                print(f"[midi] {e}")
            time.sleep(1.0)   # brief pause before retry

    def _choose_port(self) -> str:
        available = mido.get_input_names()
        if not available:
            return ""
        if self._device:
            for n in available:
                if self._device.lower() in n.lower():
                    return n
        # Fall back to auto-detect Akai / MPK.
        for n in available:
            if any(k in n.lower() for k in ("mpk", "akai")):
                return n
        return available[0]

    def _on_msg(self, msg):
        """Called by the mido/rtmidi internal thread for every incoming message."""
        ch = msg.channel + 1   # mido uses 0-indexed channels; we use 1-indexed
        if msg.type == "note_on":
            self.note_on.emit(ch, msg.note, msg.velocity)
        elif msg.type == "note_off":
            self.note_off.emit(ch, msg.note)

    def stop_listener(self):
        self._alive = False
        self.wait(2000)


# ── Pad widget ────────────────────────────────────────────────────────────────

class PadWidget(QFrame):
    """
    One cell of the 4×4 soundboard grid.
    Accepts drag-and-drop of audio files and flashes on MIDI trigger.
    """

    file_dropped = pyqtSignal(int, str)   # (pad_index, absolute_file_path)

    _C_EMPTY  = "#202030"
    _C_LOADED = "#1a3326"
    _C_FLASH  = "#4a7aff"
    _C_DRAG   = "#2a3a5a"

    _AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".aiff", ".aif"}

    def __init__(self, index: int):
        super().__init__()
        self.index     = index
        self.file_path = ""

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._dim)

        self.setAcceptDrops(True)
        self.setMinimumSize(96, 72)
        self.setFrameStyle(QFrame.Shape.Box)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(5, 5, 5, 5)
        lay.setSpacing(2)

        self._num  = QLabel(f"Pad {index + 1}")
        self._num.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._num.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))

        self._name = QLabel("drop file")
        self._name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name.setWordWrap(True)
        self._name.setFont(QFont("Segoe UI", 7))

        lay.addWidget(self._num)
        lay.addWidget(self._name)

        self._apply_bg(self._C_EMPTY)

    def _apply_bg(self, color: str):
        self.setStyleSheet(
            f"PadWidget  {{ background:{color}; border:2px solid #404060; border-radius:6px; }}"
            "QLabel      { color:#bbbbdd; background:transparent; }"
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def flash(self):
        """Brief blue flash to confirm the pad was triggered."""
        self._apply_bg(self._C_FLASH)
        self._timer.start(140)

    def set_file(self, path: str):
        self.file_path = path
        stem = Path(path).stem if path else ""
        display = (stem[:12] + "…") if len(stem) > 13 else (stem or "drop file")
        self._name.setText(display)
        self._dim()

    def _dim(self):
        self._apply_bg(self._C_LOADED if self.file_path else self._C_EMPTY)

    # ── Drag-and-drop ──────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            if any(Path(u.toLocalFile()).suffix.lower() in self._AUDIO_EXTS
                   for u in event.mimeData().urls()):
                event.acceptProposedAction()
                self._apply_bg(self._C_DRAG)

    def dragLeaveEvent(self, _event):
        self._dim()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in self._AUDIO_EXTS:
                self.set_file(path)
                self.file_dropped.emit(self.index, path)
                break
        event.acceptProposedAction()


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.cfg   = Config()
        self.audio = AudioEngine()
        self.cache = NoteCache()
        self.midi  = MidiListener()
        self.pads: list[PadWidget] = []

        self._build_ui()
        self._populate_devices()     # fill combos (no signal connections yet)
        self._restore_saved_state()  # apply saved pad files + device names
        self._start_audio()
        self._start_midi()

        # Connect device-change signals AFTER initial population to avoid
        # spurious restarts during startup.
        self.midi_cb.currentTextChanged.connect(self._on_midi_changed)
        self.audio_cb.currentTextChanged.connect(self._on_audio_changed)

        # Cross-thread MIDI signals land on the GUI thread via Qt queued connections.
        self.midi.note_on.connect(self._on_note_on)
        self.midi.note_off.connect(self._on_note_off)

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("MIDI Piano & Soundboard")
        self.setMinimumSize(600, 500)
        self.setStyleSheet("""
            * { font-family: 'Segoe UI', sans-serif; }
            QMainWindow, QWidget { background:#14142a; color:#ccccee; }
            QComboBox {
                background:#22223a; border:1px solid #444466; border-radius:4px;
                padding:4px 8px; color:#ccccee; min-width:175px;
            }
            QComboBox::drop-down { border:none; }
            QComboBox QAbstractItemView {
                background:#22223a; color:#ccccee;
                selection-background-color:#3c3c5c;
            }
            QPushButton {
                background:#22223a; border:1px solid #505070;
                border-radius:4px; padding:5px 14px; color:#aaaacc;
            }
            QPushButton:hover { background:#2e2e4e; }
            QLabel { color:#aaaacc; }
        """)

        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(14, 14, 14, 10)
        vbox.setSpacing(10)

        # ── Title ──────────────────────────────────────────────────────────
        title = QLabel("MIDI Piano & Soundboard")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title.setStyleSheet("color:#7777ff;")
        vbox.addWidget(title)

        # ── Device selectors ───────────────────────────────────────────────
        drow = QHBoxLayout()
        drow.setSpacing(8)
        drow.addWidget(QLabel("MIDI in:"))
        self.midi_cb = QComboBox()
        drow.addWidget(self.midi_cb)
        drow.addSpacing(10)
        drow.addWidget(QLabel("Audio out:"))
        self.audio_cb = QComboBox()
        drow.addWidget(self.audio_cb)
        drow.addSpacing(6)
        refresh = QPushButton("⟳  Refresh")
        refresh.clicked.connect(self._on_refresh)
        drow.addWidget(refresh)
        drow.addStretch()
        vbox.addLayout(drow)

        # ── Status line ────────────────────────────────────────────────────
        self.status = QLabel("Starting…")
        self.status.setStyleSheet("color:#555577; font-size:10px;")
        vbox.addWidget(self.status)

        # ── Pad grid header ────────────────────────────────────────────────
        hint_top = QLabel("Soundboard  —  drag MP3 / WAV files onto pads")
        hint_top.setStyleSheet("color:#555577; font-size:10px;")
        vbox.addWidget(hint_top)

        # ── 2 × 4 pad grid (matches MPK Mini MkII physical layout) ────────
        grid = QGridLayout()
        grid.setSpacing(6)
        for i in range(len(PAD_NOTES)):
            pad = PadWidget(i)
            pad.file_dropped.connect(self._on_pad_drop)
            self.pads.append(pad)
            grid.addWidget(pad, i // PAD_COLS, i % PAD_COLS)
        vbox.addLayout(grid)

        # ── Footer ─────────────────────────────────────────────────────────
        hint_bot = QLabel("Ch 1 → piano keys  ·  Ch 10 → drum pads  ·  works while minimised")
        hint_bot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint_bot.setStyleSheet("color:#383858; font-size:9px;")
        vbox.addWidget(hint_bot)

    # ── Device enumeration ─────────────────────────────────────────────────

    def _populate_devices(self):
        """Fill the MIDI and audio dropdowns without triggering change signals."""
        # ── MIDI inputs ────────────────────────────────────────────────────
        self.midi_cb.blockSignals(True)
        self.midi_cb.clear()
        for name in mido.get_input_names():
            self.midi_cb.addItem(name)
        self.midi_cb.blockSignals(False)

        # ── Audio outputs ──────────────────────────────────────────────────
        self.audio_cb.blockSignals(True)
        self.audio_cb.clear()
        try:
            for idx, dev in enumerate(sd.query_devices()):
                if dev["max_output_channels"] > 0:
                    # Store the PortAudio device index as item data.
                    self.audio_cb.addItem(dev["name"], userData=idx)
        except Exception as e:
            self.audio_cb.addItem(f"Error enumerating devices: {e}", userData=None)
        self.audio_cb.blockSignals(False)

    def _restore_saved_state(self):
        """Apply saved device choices and pad file assignments to the UI."""
        # Restore MIDI device selection.
        if self.cfg.midi_device:
            for i in range(self.midi_cb.count()):
                if self.cfg.midi_device.lower() in self.midi_cb.itemText(i).lower():
                    self.midi_cb.setCurrentIndex(i)
                    break
        else:
            # Auto-select the first Akai / MPK port found.
            for i in range(self.midi_cb.count()):
                txt = self.midi_cb.itemText(i).lower()
                if "mpk" in txt or "akai" in txt:
                    self.midi_cb.setCurrentIndex(i)
                    break

        # Restore audio device selection.
        if self.cfg.audio_device:
            for i in range(self.audio_cb.count()):
                if self.cfg.audio_device.lower() in self.audio_cb.itemText(i).lower():
                    self.audio_cb.setCurrentIndex(i)
                    break

        # Restore pad file assignments (only if the file still exists on disk).
        for pad_idx, path in self.cfg.pad_files.items():
            if 0 <= pad_idx < 16 and Path(path).exists():
                self.pads[pad_idx].set_file(path)

    # ── Startup helpers ────────────────────────────────────────────────────

    def _start_audio(self):
        try:
            self.audio.start(self.audio_cb.currentData())
            self.status.setText("Warming up piano samples…")
            threading.Thread(target=self._warm_up_worker, daemon=True).start()
        except Exception as e:
            self.status.setText(f"Audio error: {e}")

    def _warm_up_worker(self):
        """
        Runs in a background thread.
        Pre-generates all 88 piano notes and pre-loads any saved pad files
        so the first keystroke or pad hit has zero loading latency.
        """
        self.cache.warm_up()
        for path in self.cfg.pad_files.values():
            if Path(path).exists():
                self.audio.preload(path)
        # Schedule the status update back on the GUI thread.
        QTimer.singleShot(0, lambda: self.status.setText("Ready  ✓"))

    def _start_midi(self):
        self.midi.set_device(self.midi_cb.currentText())
        self.midi.start()

    # ── Slot handlers ──────────────────────────────────────────────────────

    def _on_refresh(self):
        """Re-enumerate devices and keep the current selection if still available."""
        saved_midi  = self.midi_cb.currentText()
        saved_audio = self.audio_cb.currentText()
        self.midi_cb.currentTextChanged.disconnect(self._on_midi_changed)
        self.audio_cb.currentTextChanged.disconnect(self._on_audio_changed)
        self._populate_devices()
        # Try to restore selections after refresh.
        for i in range(self.midi_cb.count()):
            if self.midi_cb.itemText(i) == saved_midi:
                self.midi_cb.setCurrentIndex(i)
                break
        for i in range(self.audio_cb.count()):
            if self.audio_cb.itemText(i) == saved_audio:
                self.audio_cb.setCurrentIndex(i)
                break
        self.midi_cb.currentTextChanged.connect(self._on_midi_changed)
        self.audio_cb.currentTextChanged.connect(self._on_audio_changed)
        self.status.setText("Devices refreshed")

    def _on_midi_changed(self, name: str):
        self.midi.set_device(name)
        self.cfg.midi_device = name
        self.cfg.save()
        self.status.setText(f"MIDI → {name}")

    def _on_audio_changed(self, name: str):
        dev_idx = self.audio_cb.currentData()
        try:
            self.audio.restart(dev_idx)
            self.cfg.audio_device = name
            self.cfg.save()
            self.status.setText(f"Audio → {name}")
        except Exception as e:
            self.status.setText(f"Audio error: {e}")

    def _on_pad_drop(self, pad_idx: int, path: str):
        self.cfg.pad_files[pad_idx] = path
        self.cfg.save()
        # Preload in the background so the next trigger is instant.
        threading.Thread(target=self.audio.preload, args=(path,), daemon=True).start()
        self.status.setText(f"Pad {pad_idx + 1} → {Path(path).name}")

    # ── MIDI dispatch ──────────────────────────────────────────────────────

    def _on_note_on(self, ch: int, note: int, vel: int):
        """
        Routes incoming NoteOn to piano or soundboard based on channel.
        This slot runs on the GUI thread (Qt queued connection).
        """
        if vel == 0:
            # Some controllers send NoteOn with velocity 0 instead of NoteOff.
            self._on_note_off(ch, note)
            return

        if ch == 1:
            # ── Piano: velocity-sensitive, polyphonic ──────────────────────
            buf  = self.cache.get(note)
            self.audio.play_note(buf, vel / 127.0)
            self.status.setText(f"Piano  {note_name(note)}  (vel {vel})")

        elif ch == 10:
            # ── Drum pads: fixed volume regardless of velocity ─────────────
            if note not in NOTE_TO_PAD:
                return
            idx = NOTE_TO_PAD[note]
            self.pads[idx].flash()
            path = self.cfg.pad_files.get(idx, "")
            if path:
                self.audio.play_pad(path)
            else:
                self.status.setText(f"Pad {idx + 1} (note {note}): no file assigned — drop a file onto it")

    def _on_note_off(self, ch: int, note: int):
        # Notes fade out via their synthesis envelope; no hard gate is needed.
        pass

    # ── Shutdown ───────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self.midi.stop_listener()
        self.audio.stop()
        self.cfg.save()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    # On Windows, tell Qt to use the system DPI so the UI is not blurry
    # on high-DPI displays.
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    app = QApplication(sys.argv)
    app.setApplicationName("MIDI Soundboard")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
