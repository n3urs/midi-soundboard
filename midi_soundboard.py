#!/usr/bin/env python3
"""
MIDI Piano & Soundboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Turns an Akai MPK Mini MKII into a real-time piano synthesizer + soundboard.

  • MIDI Ch 1  → piano synth (velocity-sensitive, polyphonic, key-release
    aware, sustain-pedal CC64 supported)
  • MIDI Ch 10 → per-pad audio file playback (MP3 / WAV / OGG / FLAC)
  • Drag audio files onto pads — mappings auto-saved to config.json
  • Right-click a pad for: choose file, clear, per-pad volume
  • Master volume slider + Stop All panic button
  • Closing the window hides to the system tray; quit from the tray menu
  • Selectable MIDI input and audio output device (VB-Audio Cable support)

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
    QSlider, QMenu, QFileDialog, QSystemTrayIcon,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QIcon, QPixmap, QPainter, QColor, QAction


# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE   = 44100   # Hz
BLOCK_SIZE    = 256     # frames per audio callback (~5.8 ms latency)
MAX_POLYPHONY = 32      # oldest voice stolen when this limit is hit

# Akai MPK Mini MkII Bank-A pad → MIDI note mapping.
# The device has 8 pads in a 2×4 grid; top row is pads 5-8, bottom is 1-4.
# Notes match this unit's factory Program 1 layout (F1–G#1 top, C#1–E1 bottom).
PAD_NOTES   = [29, 30, 31, 32,   # row 0 (top)    – pads 5-8  (F1, F#1, G1, G#1)
               25, 26, 27, 28]   # row 1 (bottom) – pads 1-4  (C#1, D1, D#1, E1)
NOTE_TO_PAD = {note: idx for idx, note in enumerate(PAD_NOTES)}
PAD_COLS    = 4   # pads per row

SUSTAIN_CC  = 64   # standard MIDI sustain-pedal controller number

# Config is stored in the user's home directory so it survives app moves.
CONFIG_FILE = Path.home() / ".midi_soundboard" / "config.json"

AUDIO_EXTS  = {".mp3", ".wav", ".ogg", ".flac", ".aiff", ".aif"}


# ── Config persistence ────────────────────────────────────────────────────────

class Config:
    """Loads and saves pad mappings + device selections to a JSON file."""

    def __init__(self):
        self.pad_files:   dict[int, str]   = {}   # pad index → file path
        self.pad_volumes: dict[int, float] = {}   # pad index → gain 0.0-1.0
        self.midi_device:  str   = ""
        self.audio_device: str   = ""
        self.master:       float = 0.9
        self._load()

    def _load(self):
        if not CONFIG_FILE.is_file():
            return
        try:
            raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            self.pad_files    = {int(k): v for k, v in raw.get("pads", {}).items()}
            self.pad_volumes  = {int(k): float(v)
                                 for k, v in raw.get("pad_volumes", {}).items()}
            self.midi_device  = raw.get("midi_device", "")
            self.audio_device = raw.get("audio_device", "")
            self.master       = float(raw.get("master", 0.9))
        except Exception:
            pass  # corrupt config → start fresh

    def save(self):
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps({
            "pads":         {str(k): v for k, v in self.pad_files.items()},
            "pad_volumes":  {str(k): v for k, v in self.pad_volumes.items()},
            "midi_device":  self.midi_device,
            "audio_device": self.audio_device,
            "master":       self.master,
        }, indent=2), encoding="utf-8")


# ── Piano synthesis ───────────────────────────────────────────────────────────

def midi_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def note_name(note: int) -> str:
    names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    return f"{names[note % 12]}{note // 12 - 1}"


def synthesise_note(freq: float) -> np.ndarray:
    """
    Piano-like tone via additive synthesis. Returns stereo float32 (N, 2).

    What makes it sound piano-ish rather than organ-ish:
      • inharmonicity — real piano strings are stiff, so upper partials run
        progressively sharp (f_h = f·h·√(1+B·h²))
      • per-partial decay — high partials die quickly, leaving a mellow tail
      • a short filtered-noise "hammer" transient at the attack
      • per-partial stereo panning for width
    Release shaping (key-up / sustain pedal) is handled live by the Voice.
    """
    duration  = float(np.clip(6.0 * (110.0 / freq) ** 0.5, 1.2, 6.0))
    n_samples = int(duration * SAMPLE_RATE)
    t         = np.linspace(0.0, duration, n_samples, dtype=np.float32)

    B     = 0.0004                       # string-stiffness coefficient
    left  = np.zeros(n_samples, dtype=np.float32)
    right = np.zeros(n_samples, dtype=np.float32)

    for h in range(1, 13):
        fh = freq * h * np.sqrt(1.0 + B * h * h)
        if fh >= SAMPLE_RATE / 2:        # Nyquist guard
            break
        amp     = 1.0 / (h ** 1.8)
        env     = np.exp(-t * (2.2 + 0.7 * h) / duration).astype(np.float32)
        partial = (amp * env *
                   np.sin(2 * np.pi * fh * t).astype(np.float32))
        # Spread partials across the stereo field for a wider, livelier tone.
        pan    = 0.5 + 0.18 * np.sin(h * 2.1 + freq * 0.01)
        left  += partial * np.float32(np.cos(pan * np.pi / 2))
        right += partial * np.float32(np.sin(pan * np.pi / 2))

    # Hammer transient: 10 ms of low-passed noise mixed into the attack.
    n_thump = int(0.010 * SAMPLE_RATE)
    rng     = np.random.default_rng(int(freq))   # deterministic per note
    noise   = rng.standard_normal(n_thump).astype(np.float32)
    noise   = np.convolve(noise, np.ones(48, dtype=np.float32) / 48,
                          mode="same").astype(np.float32)
    noise  *= np.linspace(1.0, 0.0, n_thump, dtype=np.float32) * 0.6
    left[:n_thump]  += noise
    right[:n_thump] += noise

    # 3 ms linear ramp removes the click at note onset.
    ramp  = max(1, int(0.003 * SAMPLE_RATE))
    onset = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
    left[:ramp]  *= onset
    right[:ramp] *= onset

    stereo = np.column_stack([left, right])
    peak   = float(np.max(np.abs(stereo)))
    if peak:
        stereo *= 0.70 / peak
    return stereo


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
            self.get(note)

    def get(self, note: int) -> np.ndarray:
        with self._lock:
            if note not in self._data:
                self._data[note] = synthesise_note(midi_to_hz(note))
            return self._data[note]


# ── Audio engine ──────────────────────────────────────────────────────────────

class Voice:
    """
    A cursor over a pre-computed stereo buffer, with an optional live
    release fade (used for key-up, retriggered pads, and Stop All).

    `tag` identifies the voice for targeted control:
        ("note", midi_note)  → piano voice
        ("pad",  pad_index)  → soundboard voice
    """
    __slots__ = ("buf", "pos", "gain", "done", "tag",
                 "_rel_total", "_rel_left")

    def __init__(self, buf: np.ndarray, gain: float = 1.0, tag=None):
        self.buf  = buf      # (N, 2) float32
        self.pos  = 0
        self.gain = np.float32(gain)
        self.done = False
        self.tag  = tag
        self._rel_total = 0   # >0 once a release has been triggered
        self._rel_left  = 0

    def start_release(self, fast: bool = False):
        """Begin fading this voice out (80 ms normally, 15 ms for cuts)."""
        if self._rel_total or self.done:
            return
        self._rel_total = int((0.015 if fast else 0.080) * SAMPLE_RATE)
        self._rel_left  = self._rel_total

    def read(self, n: int) -> np.ndarray:
        if self.done:
            return np.zeros((n, 2), dtype=np.float32)
        chunk     = self.buf[self.pos : self.pos + n]
        self.pos += n
        if self.pos >= len(self.buf):
            self.done = True
        if len(chunk) < n:
            chunk = np.pad(chunk, ((0, n - len(chunk)), (0, 0)))
        else:
            chunk = chunk.copy()          # never scale the shared cached buffer

        if self._rel_total:
            # Linear fade from the current release position down to silence.
            r0 = self._rel_left / self._rel_total
            r1 = max(0.0, (self._rel_left - n) / self._rel_total)
            chunk *= np.linspace(r0, r1, n, dtype=np.float32)[:, None]
            self._rel_left -= n
            if self._rel_left <= 0:
                self.done = True
        return chunk * self.gain


class AudioEngine:
    """
    Owns a low-latency PortAudio output stream via sounddevice.
    All public methods are thread-safe; they are called from the MIDI thread.
    """

    def __init__(self, master: float = 0.9):
        self._voices : list[Voice]           = []
        self._samples: dict[str, np.ndarray] = {}   # path → (N,2) float32
        self._lock    = threading.Lock()
        self._stream : sd.OutputStream | None = None
        self._master  = np.float32(master)
        self._sustain = False
        self._deferred: set[int] = set()   # notes released while pedal down

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
        mix *= self._master
        # Hard-clip to [-1, 1] prevents distortion when many voices overlap.
        np.clip(mix, -1.0, 1.0, out=out)

    # ── Public play API ────────────────────────────────────────────────────

    def play_note(self, buf: np.ndarray, velocity: float, note: int):
        """Start a piano voice. Restriking a ringing note cuts the old one."""
        voice = Voice(buf, velocity * 0.8, tag=("note", note))
        with self._lock:
            for v in self._voices:
                if v.tag == ("note", note):
                    v.start_release(fast=True)
            self._deferred.discard(note)
            if len(self._voices) >= MAX_POLYPHONY:
                self._voices.pop(0)           # steal the oldest voice
            self._voices.append(voice)

    def note_off(self, note: int):
        """Key released: fade the note out, unless the sustain pedal is down."""
        with self._lock:
            if self._sustain:
                self._deferred.add(note)
                return
            for v in self._voices:
                if v.tag == ("note", note):
                    v.start_release()

    def set_sustain(self, down: bool):
        """CC64 sustain pedal. Releasing it fades all deferred notes."""
        with self._lock:
            self._sustain = down
            if not down:
                for v in self._voices:
                    if v.tag and v.tag[0] == "note" and v.tag[1] in self._deferred:
                        v.start_release()
                self._deferred.clear()

    def play_pad(self, path: str, pad_idx: int, gain: float = 1.0):
        """Trigger a soundboard sample; retriggering cuts the previous play."""
        buf = self._load(path)
        if buf is None:
            return
        with self._lock:
            for v in self._voices:
                if v.tag == ("pad", pad_idx):
                    v.start_release(fast=True)
            self._voices.append(Voice(buf, gain, tag=("pad", pad_idx)))

    def stop_all(self):
        """Panic button: quick-fade every active voice."""
        with self._lock:
            for v in self._voices:
                v.start_release(fast=True)
            self._deferred.clear()

    def set_master(self, gain: float):
        self._master = np.float32(gain)

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
    cc       = pyqtSignal(int, int, int)   # channel, controller, value

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
        if not hasattr(msg, "channel"):
            return
        ch = msg.channel + 1   # mido uses 0-indexed channels; we use 1-indexed
        if msg.type == "note_on":
            self.note_on.emit(ch, msg.note, msg.velocity)
        elif msg.type == "note_off":
            self.note_off.emit(ch, msg.note)
        elif msg.type == "control_change":
            self.cc.emit(ch, msg.control, msg.value)

    def stop_listener(self):
        self._alive = False
        self.wait(2000)


# ── Pad widget ────────────────────────────────────────────────────────────────

class PadWidget(QFrame):
    """
    One cell of the 2×4 soundboard grid.
    Drag-and-drop to assign a file; right-click for choose / clear / volume.
    Flashes on MIDI trigger.
    """

    file_dropped   = pyqtSignal(int, str)     # (pad_index, file_path)
    cleared        = pyqtSignal(int)          # (pad_index)
    volume_changed = pyqtSignal(int, float)   # (pad_index, gain 0.0-1.0)

    # Visual states: (gradient top, gradient bottom, border, name colour)
    _STATES = {
        "empty":  ("#1a1e2b", "#151823", "#262c3d", "#4d5675"),
        "loaded": ("#1e2a44", "#182032", "#3a4c78", "#d6def3"),
        "flash":  ("#6c8cff", "#4a6ee8", "#9db4ff", "#ffffff"),
        "drag":   ("#26365c", "#1e2a48", "#6c8cff", "#aebcf0"),
    }

    def __init__(self, index: int):
        super().__init__()
        self.index     = index
        self.file_path = ""
        self.volume    = 1.0

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._dim)

        self.setAcceptDrops(True)
        self.setMinimumSize(124, 96)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(3)

        self._num = QLabel(f"PAD {index + 1}")
        self._num.setObjectName("padNum")
        self._num.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self._name = QLabel("drop a file")
        self._name.setObjectName("padName")
        self._name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name.setWordWrap(True)

        self._vol = QLabel("")
        self._vol.setObjectName("padVol")
        self._vol.setAlignment(Qt.AlignmentFlag.AlignRight)

        lay.addWidget(self._num)
        lay.addStretch()
        lay.addWidget(self._name)
        lay.addStretch()
        lay.addWidget(self._vol)

        self._set_state("empty")

    def _set_state(self, state: str):
        top, bottom, border, name_col = self._STATES[state]
        self.setStyleSheet(f"""
            PadWidget {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 {top}, stop:1 {bottom});
                border: 1px solid {border};
                border-radius: 10px;
            }}
            QLabel          {{ background: transparent; border: none; }}
            QLabel#padNum   {{ color: #5d6790; font-size: 8pt; font-weight: 600;
                               letter-spacing: 1px; }}
            QLabel#padName  {{ color: {name_col}; font-size: 9pt; font-weight: 500; }}
            QLabel#padVol   {{ color: #5d6790; font-size: 7pt; }}
        """)

    # ── Public API ─────────────────────────────────────────────────────────

    def flash(self):
        """Brief blue flash to confirm the pad was triggered."""
        self._set_state("flash")
        self._timer.start(140)

    def set_file(self, path: str):
        self.file_path = path
        stem = Path(path).stem if path else ""
        display = (stem[:16] + "…") if len(stem) > 17 else (stem or "drop a file")
        self._name.setText(display)
        self._dim()

    def set_volume(self, gain: float):
        self.volume = gain
        self._vol.setText(f"{int(gain * 100)}%" if gain < 1.0 else "")

    def _dim(self):
        self._set_state("loaded" if self.file_path else "empty")

    # ── Right-click menu ───────────────────────────────────────────────────

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        act_choose = menu.addAction("Choose file…")
        act_clear  = menu.addAction("Clear file")
        act_clear.setEnabled(bool(self.file_path))
        vol_menu   = menu.addMenu("Volume")
        for pct in (25, 50, 75, 100):
            a = QAction(f"{pct}%", vol_menu)
            a.setCheckable(True)
            a.setChecked(abs(self.volume - pct / 100) < 0.01)
            a.setData(pct / 100)
            vol_menu.addAction(a)

        chosen = menu.exec(event.globalPos())
        if chosen is act_choose:
            exts = " ".join(f"*{e}" for e in sorted(AUDIO_EXTS))
            path, _ = QFileDialog.getOpenFileName(
                self, "Choose audio file", "", f"Audio files ({exts})")
            if path:
                self.set_file(path)
                self.file_dropped.emit(self.index, path)
        elif chosen is act_clear:
            self.set_file("")
            self.cleared.emit(self.index)
        elif chosen is not None and chosen.data() is not None:
            self.set_volume(float(chosen.data()))
            self.volume_changed.emit(self.index, self.volume)

    # ── Drag-and-drop ──────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            if any(Path(u.toLocalFile()).suffix.lower() in AUDIO_EXTS
                   for u in event.mimeData().urls()):
                event.acceptProposedAction()
                self._set_state("drag")

    def dragLeaveEvent(self, _event):
        self._dim()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in AUDIO_EXTS:
                self.set_file(path)
                self.file_dropped.emit(self.index, path)
                break
        event.acceptProposedAction()


# ── Main window ───────────────────────────────────────────────────────────────

def _make_app_icon() -> QIcon:
    """Draw a simple ♪ tile so the tray/taskbar has an icon without assets."""
    pm = QPixmap(64, 64)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor("#4a7aff"))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(2, 2, 60, 60, 14, 14)
    p.setPen(QColor("white"))
    p.setFont(QFont("Segoe UI", 34, QFont.Weight.Bold))
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "♪")
    p.end()
    return QIcon(pm)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.cfg   = Config()
        self.audio = AudioEngine(master=self.cfg.master)
        self.cache = NoteCache()
        self.midi  = MidiListener()
        self.pads: list[PadWidget] = []
        self._quitting    = False
        self._tray_warned = False

        self._build_ui()
        self._build_tray()
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
        self.midi.cc.connect(self._on_cc)

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("MIDI Piano & Soundboard")
        self.setWindowIcon(_make_app_icon())
        self.setMinimumSize(640, 500)
        self.setStyleSheet("""
            * { font-family: 'Segoe UI', 'SF Pro Text', -apple-system, sans-serif; }
            QMainWindow, QWidget { background:#101218; color:#c9d1e8; }
            QFrame#card {
                background:#171a24; border:1px solid #232838; border-radius:12px;
            }
            QLabel#cardTitle {
                color:#5d6790; font-size:8pt; font-weight:600; letter-spacing:2px;
            }
            QLabel#fieldLabel { color:#7b86ad; font-size:9pt; }
            QComboBox {
                background:#1d2230; border:1px solid #2c3347; border-radius:6px;
                padding:5px 10px; color:#c9d1e8; min-width:175px; font-size:9pt;
            }
            QComboBox:hover { border-color:#3d4763; }
            QComboBox::drop-down { border:none; width:22px; }
            QComboBox QAbstractItemView {
                background:#1d2230; color:#c9d1e8; border:1px solid #2c3347;
                selection-background-color:#2c3854; outline:none;
            }
            QPushButton {
                background:#1d2230; border:1px solid #2c3347; border-radius:6px;
                padding:6px 16px; color:#aeb8d6; font-size:9pt;
            }
            QPushButton:hover { background:#252b3d; border-color:#6c8cff; color:#e3e8f7; }
            QPushButton:pressed { background:#1a1f2e; }
            QPushButton#stopBtn:hover { border-color:#ff6c7a; color:#ffb3bb; }
            QLabel { color:#aeb8d6; }
            QSlider::groove:horizontal {
                height:4px; background:#252b3d; border-radius:2px;
            }
            QSlider::sub-page:horizontal { background:#6c8cff; border-radius:2px; }
            QSlider::handle:horizontal {
                width:14px; margin:-5px 0; border-radius:7px; background:#aebcf0;
            }
            QSlider::handle:horizontal:hover { background:#d4ddfa; }
            QMenu {
                background:#1d2230; color:#c9d1e8; border:1px solid #2c3347;
                border-radius:8px; padding:4px;
            }
            QMenu::item { padding:5px 22px; border-radius:4px; }
            QMenu::item:selected { background:#2c3854; }
        """)

        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(18, 16, 18, 12)
        vbox.setSpacing(12)

        # ── Header: title + live status on one line ────────────────────────
        hrow = QHBoxLayout()
        title = QLabel("MIDI Piano & Soundboard")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color:#e3e8f7;")
        subtitle = QLabel("Akai MPK Mini MkII")
        subtitle.setStyleSheet("color:#5d6790; font-size:9pt; padding-top:4px;")
        hrow.addWidget(title)
        hrow.addSpacing(8)
        hrow.addWidget(subtitle)
        hrow.addStretch()
        self.status = QLabel("Starting…")
        self.status.setStyleSheet("color:#6c8cff; font-size:9pt;")
        self.status.setAlignment(Qt.AlignmentFlag.AlignRight |
                                 Qt.AlignmentFlag.AlignVCenter)
        hrow.addWidget(self.status)
        vbox.addLayout(hrow)

        # ── Card: devices + volume ─────────────────────────────────────────
        card = QFrame()
        card.setObjectName("card")
        clay = QVBoxLayout(card)
        clay.setContentsMargins(16, 12, 16, 14)
        clay.setSpacing(10)

        drow = QHBoxLayout()
        drow.setSpacing(8)
        lbl_midi = QLabel("MIDI in")
        lbl_midi.setObjectName("fieldLabel")
        drow.addWidget(lbl_midi)
        self.midi_cb = QComboBox()
        drow.addWidget(self.midi_cb, 1)
        drow.addSpacing(12)
        lbl_audio = QLabel("Audio out")
        lbl_audio.setObjectName("fieldLabel")
        drow.addWidget(lbl_audio)
        self.audio_cb = QComboBox()
        drow.addWidget(self.audio_cb, 1)
        drow.addSpacing(8)
        refresh = QPushButton("⟳  Refresh")
        refresh.clicked.connect(self._on_refresh)
        drow.addWidget(refresh)
        clay.addLayout(drow)

        vrow = QHBoxLayout()
        vrow.setSpacing(8)
        lbl_vol = QLabel("Volume")
        lbl_vol.setObjectName("fieldLabel")
        vrow.addWidget(lbl_vol)
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(int(self.cfg.master * 100))
        self.vol_slider.setMaximumWidth(240)
        self.vol_slider.valueChanged.connect(self._on_master_changed)
        vrow.addWidget(self.vol_slider)
        vrow.addStretch()
        stop_btn = QPushButton("⏹  Stop All")
        stop_btn.setObjectName("stopBtn")
        stop_btn.clicked.connect(self._on_stop_all)
        vrow.addWidget(stop_btn)
        clay.addLayout(vrow)

        vbox.addWidget(card)

        # ── Soundboard section ─────────────────────────────────────────────
        srow = QHBoxLayout()
        sb_title = QLabel("SOUNDBOARD")
        sb_title.setObjectName("cardTitle")
        srow.addWidget(sb_title)
        srow.addStretch()
        sb_hint = QLabel("drag files onto pads · right-click for options")
        sb_hint.setStyleSheet("color:#444d6e; font-size:8pt;")
        srow.addWidget(sb_hint)
        vbox.addLayout(srow)

        # 2 × 4 pad grid (matches MPK Mini MkII physical layout)
        grid = QGridLayout()
        grid.setSpacing(10)
        for i in range(len(PAD_NOTES)):
            pad = PadWidget(i)
            pad.file_dropped.connect(self._on_pad_drop)
            pad.cleared.connect(self._on_pad_clear)
            pad.volume_changed.connect(self._on_pad_volume)
            self.pads.append(pad)
            grid.addWidget(pad, i // PAD_COLS, i % PAD_COLS)
        vbox.addLayout(grid, 1)

        # ── Footer ─────────────────────────────────────────────────────────
        hint_bot = QLabel("ch 1 → piano keys   ·   ch 10 → drum pads   ·   closing hides to tray")
        hint_bot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint_bot.setStyleSheet("color:#363e5c; font-size:8pt;")
        vbox.addWidget(hint_bot)

    def _build_tray(self):
        """System-tray icon so the app keeps running when the window closes."""
        self.tray = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self.tray = QSystemTrayIcon(_make_app_icon(), self)
        menu = QMenu()
        act_show = menu.addAction("Show window")
        act_show.triggered.connect(self._tray_show)
        act_quit = menu.addAction("Quit")
        act_quit.triggered.connect(self._tray_quit)
        self.tray.setContextMenu(menu)
        self._tray_menu = menu          # keep a reference so Qt doesn't GC it
        self.tray.setToolTip("MIDI Piano & Soundboard")
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._tray_show()

    def _tray_show(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _tray_quit(self):
        self._quitting = True
        self.close()

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

        # Restore pad assignments (only if the file still exists on disk).
        for pad_idx, path in self.cfg.pad_files.items():
            if 0 <= pad_idx < len(self.pads) and Path(path).exists():
                self.pads[pad_idx].set_file(path)
        for pad_idx, gain in self.cfg.pad_volumes.items():
            if 0 <= pad_idx < len(self.pads):
                self.pads[pad_idx].set_volume(gain)

    # ── Startup helpers ────────────────────────────────────────────────────

    def _start_audio(self):
        try:
            self.audio.start(self.audio_cb.currentData())
            self.status.setText("Warming up piano notes…")
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

    def _on_master_changed(self, value: int):
        self.audio.set_master(value / 100)
        self.cfg.master = value / 100
        self.cfg.save()

    def _on_stop_all(self):
        self.audio.stop_all()
        self.status.setText("Stopped all sounds")

    def _on_pad_drop(self, pad_idx: int, path: str):
        self.cfg.pad_files[pad_idx] = path
        self.cfg.save()
        # Preload in the background so the next trigger is instant.
        threading.Thread(target=self.audio.preload, args=(path,), daemon=True).start()
        self.status.setText(f"Pad {pad_idx + 1} → {Path(path).name}")

    def _on_pad_clear(self, pad_idx: int):
        self.cfg.pad_files.pop(pad_idx, None)
        self.cfg.save()
        self.status.setText(f"Pad {pad_idx + 1} cleared")

    def _on_pad_volume(self, pad_idx: int, gain: float):
        self.cfg.pad_volumes[pad_idx] = gain
        self.cfg.save()
        self.status.setText(f"Pad {pad_idx + 1} volume → {int(gain * 100)}%")

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
            buf = self.cache.get(note)
            self.audio.play_note(buf, vel / 127.0, note)
            self.status.setText(f"Piano  {note_name(note)}  (vel {vel})")

        elif ch == 10:
            # ── Drum pads: fixed volume regardless of velocity ─────────────
            if note not in NOTE_TO_PAD:
                self.status.setText(
                    f"Pad note {note} not in Bank A mapping — is the pad bank set to A?")
                return
            idx = NOTE_TO_PAD[note]
            self.pads[idx].flash()
            path = self.cfg.pad_files.get(idx, "")
            if path:
                self.audio.play_pad(path, idx, self.cfg.pad_volumes.get(idx, 1.0))
            else:
                self.status.setText(
                    f"Pad {idx + 1} (note {note}): no file assigned — drop a file onto it")

        else:
            # Anything else (e.g. pads switched to a different channel) is
            # surfaced so misconfiguration is visible instead of silent.
            self.status.setText(f"Note {note_name(note)} on ch {ch} (unrouted)")

    def _on_note_off(self, ch: int, note: int):
        if ch == 1:
            self.audio.note_off(note)

    def _on_cc(self, ch: int, control: int, value: int):
        """Knobs and pedals. CC64 = sustain; everything else just shows status."""
        if control == SUSTAIN_CC:
            self.audio.set_sustain(value >= 64)
            self.status.setText(f"Sustain {'on' if value >= 64 else 'off'}")
        else:
            self.status.setText(f"Knob CC{control} = {value} (ch {ch}, unmapped)")

    # ── Shutdown ───────────────────────────────────────────────────────────

    def closeEvent(self, event):
        # Close button hides to the tray so MIDI keeps working in-game.
        if self.tray and not self._quitting:
            event.ignore()
            self.hide()
            if not self._tray_warned:
                self._tray_warned = True
                self.tray.showMessage(
                    "Still running",
                    "MIDI Soundboard is in the system tray. Right-click the icon to quit.",
                    QSystemTrayIcon.MessageIcon.Information, 4000)
            return
        self.midi.stop_listener()
        self.audio.stop()
        self.cfg.save()
        if self.tray:
            self.tray.hide()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    # On Windows, tell Qt to use the system DPI so the UI is not blurry
    # on high-DPI displays.
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    app = QApplication(sys.argv)
    app.setApplicationName("MIDI Soundboard")
    app.setWindowIcon(_make_app_icon())
    # Keep running when the window is hidden to the tray.
    app.setQuitOnLastWindowClosed(False)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
