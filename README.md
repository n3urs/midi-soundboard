# MIDI Piano & Soundboard

Turns an **Akai MPK Mini MkII** into a real-time piano synthesizer and
soundboard — built for playing sounds into games (e.g. via VB-Audio
Virtual Cable into Overwatch voice chat).

## Features

- **Piano** (MIDI channel 1): velocity-sensitive polyphonic synth with
  inharmonic partials, hammer-noise attack, key-release fades and
  sustain-pedal (CC64) support
- **Soundboard** (MIDI channel 10): 8 pads matching the MPK Mini layout —
  drag MP3/WAV/OGG/FLAC files onto pads, or right-click to choose a file,
  clear it, or set per-pad volume
- Retriggering a pad cuts the previous playback (no stacking)
- Master volume slider + **Stop All** panic button
- Pad mappings, volumes and device choices auto-saved to
  `~/.midi_soundboard/config.json`
- Closing the window hides to the **system tray** — keeps working while
  you game; quit from the tray menu
- Selectable MIDI input and audio output device (pick *CABLE Input
  (VB-Audio Virtual Cable)* to route into voice chat)

## Run from source

```bat
pip install -r requirements.txt
python midi_soundboard.py
```

## Build a standalone Windows .exe

```bat
pip install pyinstaller

pyinstaller --noconfirm --onefile --windowed --name "MIDI_Soundboard" ^
  --collect-all sounddevice --collect-all soundfile ^
  --hidden-import mido.backends.rtmidi --hidden-import rtmidi ^
  midi_soundboard.py
```

The executable lands in `dist\MIDI_Soundboard.exe`.

## Notes

- Pads must be on **Bank A** (notes 36–43) and send on channel 10 — the
  MPK Mini MkII factory default. If a pad hit shows "not in Bank A
  mapping" in the status line, check the bank button.
- MP3 loading needs `soundfile >= 0.12.1`. If an MP3 fails, convert it to
  WAV (`ffmpeg -i clip.mp3 clip.wav`).
