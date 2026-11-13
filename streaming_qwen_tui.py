# ============================================================
# Offline mode: configure before importing Hugging Face libraries
# ============================================================
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ============================================================
# Imports
# ============================================================
import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
import webrtcvad

import arabic_reshaper
from bidi.algorithm import get_display

from qwen_asr import Qwen3ASRModel

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ============================================================
# Paths and model
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = SCRIPT_DIR / "models" / "Taraneh-Q0.6B"
LOG_FILE = SCRIPT_DIR / "transcriptions.txt"

if (LOCAL_MODEL_DIR / "config.json").is_file():
    MODEL_ID = str(LOCAL_MODEL_DIR)
else:
    # Offline mode: this works only if this repository is already cached.
    MODEL_ID = "areffarhadi/Taraneh-Q0.6B"


# ============================================================
# Audio configuration
# ============================================================
INPUT_DEVICE = 4
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1024

FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

# ============================================================
# Medium-sensitivity VAD / noise settings
# ============================================================
VAD_MODE = 2
VAD_GAIN = 1.0

START_VOICED_FRAMES = 6
PRE_ROLL_MS = 350
END_SILENCE_MS = 700
KEEP_TAIL_MS = 220

MIN_FRAME_RMS = 0.0018
NOISE_MULTIPLIER = 1.8
INITIAL_NOISE_RMS = 0.0008
NOISE_UPDATE_ALPHA = 0.02

MIN_VOICED_MS = 260
MIN_UTTERANCE_SECONDS = 0.60
MIN_VOICED_RATIO = 0.45
MAX_UTTERANCE_SECONDS = 20

MIN_RMS = 0.0021
MIN_PEAK = 0.0085
MAX_WORDS_PER_SECOND = 8.0

CPU_THREADS = min(4, os.cpu_count() or 2)

# Bounded queues prevent unlimited memory growth.
audio_queue = queue.Queue(maxsize=2000)
asr_queue = queue.Queue(maxsize=8)
status_queue = queue.Queue(maxsize=32)

callback_drop_count = 0
callback_drop_lock = threading.Lock()

console = Console()


# ============================================================
# Persian / RTL terminal rendering
# ============================================================
def rtl_display(text: str) -> str:
    """Shape Persian letters and reorder them for LTR-only terminals.

    This transformation is ONLY for terminal display. Original text is
    written to the log file in normal Unicode logical order.
    """
    if not text:
        return ""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped, base_dir="R")


# ============================================================
# Shared UI state
# ============================================================
state_lock = threading.Lock()
ui_state = {
    "speech_active": False,
    "decoding": False,
    "mic_rms": 0.0,
    "mic_peak": 0.0,
    "noise_rms": INITIAL_NOISE_RMS,
    "utterance_count": 0,
    "ignored_count": 0,
    "last_event": "Starting",
    "last_decode": 0.0,
    "last_rtf": 0.0,
    "last_queue_seconds": 0.0,
    "dropped_blocks": 0,
    "history": deque(maxlen=5),
    "error": "",
}


def update_state(**kwargs):
    with state_lock:
        ui_state.update(kwargs)


def add_history(timestamp: str, text: str, accepted: bool):
    with state_lock:
        ui_state["history"].append((timestamp, text, accepted))


# ============================================================
# Logging
# ============================================================
def append_transcription_log(
    text,
    number,
    duration,
    rms,
    peak,
    voiced_ratio,
    accepted,
    reason="",
    decode_time=0.0,
    language="Persian",
):
    """Append every non-empty ASR output; preserve natural Unicode order."""
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    status = "ACCEPTED" if accepted else "REJECTED"

    header = (
        f"[{timestamp}] [Utterance={number}] [{status}] "
        f"[Language={language}] [Duration={duration:.2f}s] "
        f"[RMS={rms:.5f}] [Peak={peak:.5f}] "
        f"[VAD={voiced_ratio:.2f}] [Decode={decode_time:.2f}s]"
    )
    if reason:
        header += f" [Reason={reason}]"

    with LOG_FILE.open("a", encoding="utf-8", newline="\n") as log:
        log.write(header + "\n")
        log.write(text + "\n")
        log.write("-" * 80 + "\n")
        log.flush()


# ============================================================
# Audio callback: keep it short; never run ASR here
# ============================================================
def audio_callback(indata, frames, time_info, status):
    global callback_drop_count

    if status:
        try:
            status_queue.put_nowait(str(status))
        except queue.Full:
            pass

    try:
        audio_queue.put_nowait(indata[:, 0].copy())
    except queue.Full:
        with callback_drop_lock:
            callback_drop_count += 1


# ============================================================
# Voice Activity Detection + adaptive noise gate
# ============================================================
class UtteranceSegmenter:
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.pre_roll_count = PRE_ROLL_MS // FRAME_MS
        self.end_silence_count = END_SILENCE_MS // FRAME_MS
        self.keep_tail_count = KEEP_TAIL_MS // FRAME_MS
        self.min_voiced_count = max(1, MIN_VOICED_MS // FRAME_MS)
        self.max_frames = int(MAX_UTTERANCE_SECONDS * 1000 / FRAME_MS)
        self.pre_roll = deque(maxlen=self.pre_roll_count)
        self.noise_rms = INITIAL_NOISE_RMS
        self.reset()

    def reset(self):
        self.active = False
        self.frames = []
        self.voiced_count = 0
        self.silence_count = 0

    def is_voiced(self, frame):
        frame_rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        vad_input = np.clip(frame * VAD_GAIN, -1.0, 1.0)
        pcm16 = (vad_input * 32767.0).astype("<i2")
        vad_result = self.vad.is_speech(pcm16.tobytes(), SAMPLE_RATE)

        # Estimate room noise only during idle, non-speech periods.
        if not self.active and not vad_result:
            self.noise_rms = (
                (1.0 - NOISE_UPDATE_ALPHA) * self.noise_rms
                + NOISE_UPDATE_ALPHA * frame_rms
            )

        threshold = max(MIN_FRAME_RMS, self.noise_rms * NOISE_MULTIPLIER)
        return vad_result and frame_rms >= threshold

    def finalize(self):
        if not self.frames:
            self.reset()
            self.pre_roll.clear()
            return "discard", "empty segment"

        trim = max(0, self.silence_count - self.keep_tail_count)
        final_frames = self.frames[:-trim] if trim else self.frames[:]

        duration = len(final_frames) * FRAME_MS / 1000.0
        voiced_ms = self.voiced_count * FRAME_MS
        voiced_ratio = self.voiced_count / max(len(final_frames), 1)

        self.reset()
        self.pre_roll.clear()

        if duration < MIN_UTTERANCE_SECONDS:
            return "discard", f"too short: {duration:.2f}s"
        if voiced_ms < MIN_VOICED_MS:
            return "discard", f"insufficient voiced audio: {voiced_ms}ms"
        if voiced_ratio < MIN_VOICED_RATIO:
            return "discard", f"low voiced ratio: {voiced_ratio:.2f}"

        return "end", (final_frames, voiced_ratio)

    def push(self, frame):
        voiced = self.is_voiced(frame)

        if not self.active:
            self.pre_roll.append((frame, voiced))
            recent_voiced = sum(1 for _, flag in self.pre_roll if flag)

            if recent_voiced >= START_VOICED_FRAMES:
                self.active = True
                self.frames = [samples for samples, _ in self.pre_roll]
                self.voiced_count = recent_voiced
                self.silence_count = 0
                return "start", None
            return None, None

        self.frames.append(frame)
        if voiced:
            self.voiced_count += 1
            self.silence_count = 0
        else:
            self.silence_count += 1

        if (
            self.silence_count >= self.end_silence_count
            or len(self.frames) >= self.max_frames
        ):
            return self.finalize()

        return None, None

    def flush(self):
        if not self.active:
            return None, None
        return self.finalize()


# ============================================================
# Text sanity check
# ============================================================
def validate_text(text: str, duration: float):
    words = text.strip().split()
    if not words:
        return False, "empty transcription"

    word_rate = len(words) / max(duration, 0.01)
    if word_rate > MAX_WORDS_PER_SECOND:
        return False, f"unrealistic word rate: {word_rate:.1f}/s"

    for n in (3, 4):
        for i in range(len(words) - 2 * n + 1):
            if words[i:i+n] == words[i+n:i+2*n]:
                return False, "repeated phrase detected"

    return True, "OK"


# ============================================================
# ASR worker: infer in background so the UI and microphone keep running
# ============================================================
def asr_worker(asr):
    while True:
        job = asr_queue.get()
        if job is None:
            asr_queue.task_done()
            break

        number, frames, voiced_ratio = job
        update_state(decoding=True, last_event=f"Decoding utterance {number}", error="")

        try:
            audio = np.concatenate(frames).astype(np.float32, copy=False)
            duration = len(audio) / SAMPLE_RATE
            audio = audio - np.mean(audio)

            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if audio.size else 0.0

            if rms < MIN_RMS:
                update_state(last_event=f"Segment {number} rejected: low RMS", ignored_count=ui_state["ignored_count"] + 1)
                continue
            if peak < MIN_PEAK:
                update_state(last_event=f"Segment {number} rejected: low peak", ignored_count=ui_state["ignored_count"] + 1)
                continue

            started = time.perf_counter()
            results = asr.transcribe(
                audio=(audio, SAMPLE_RATE),
                language="Persian",
            )
            decode_time = time.perf_counter() - started

            if not results:
                update_state(last_event=f"No result for utterance {number}")
                continue

            result = results[0]
            text = str(getattr(result, "text", "") or "").strip()
            language = str(getattr(result, "language", "Persian") or "Persian")

            if not text:
                update_state(last_event=f"Empty result for utterance {number}")
                continue

            accepted, reason = validate_text(text, duration)
            append_transcription_log(
                text=text,
                number=number,
                duration=duration,
                rms=rms,
                peak=peak,
                voiced_ratio=voiced_ratio,
                accepted=accepted,
                reason="" if accepted else reason,
                decode_time=decode_time,
                language=language,
            )

            stamp = datetime.now().astimezone().strftime("%H:%M:%S")
            add_history(stamp, text, accepted)

            rtf = decode_time / max(duration, 0.001)
            queued_seconds = audio_queue.qsize() * BLOCK_SIZE / SAMPLE_RATE

            update_state(
                utterance_count=number,
                last_event=("Transcription accepted" if accepted else f"Text flagged: {reason}"),
                last_decode=decode_time,
                last_rtf=rtf,
                last_queue_seconds=queued_seconds,
            )

        except Exception as exc:
            update_state(
                error=f"{type(exc).__name__}: {exc}",
                last_event="ASR error",
            )
        finally:
            update_state(decoding=False)
            asr_queue.task_done()


# ============================================================
# Rich terminal dashboard
# ============================================================
def build_dashboard():
    with state_lock:
        snapshot = {
            "speech_active": ui_state["speech_active"],
            "decoding": ui_state["decoding"],
            "mic_rms": ui_state["mic_rms"],
            "mic_peak": ui_state["mic_peak"],
            "noise_rms": ui_state["noise_rms"],
            "utterance_count": ui_state["utterance_count"],
            "ignored_count": ui_state["ignored_count"],
            "last_event": ui_state["last_event"],
            "last_decode": ui_state["last_decode"],
            "last_rtf": ui_state["last_rtf"],
            "last_queue_seconds": ui_state["last_queue_seconds"],
            "dropped_blocks": ui_state["dropped_blocks"],
            "history": list(ui_state["history"]),
            "error": ui_state["error"],
        }

    spinner_frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    spinner = spinner_frames[int(time.monotonic() * 8) % len(spinner_frames)]

    if snapshot["decoding"]:
        status_text = f"{spinner}  DECODING"
        status_style = "bold yellow"
    elif snapshot["speech_active"]:
        status_text = f"{spinner}  SPEECH DETECTED"
        status_style = "bold green"
    else:
        status_text = f"{spinner}  LISTENING"
        status_style = "bold bright_green"

    title = Text("TARANEH  /  LIVE SPEECH RECOGNITION", style="bold bright_cyan", justify="center")
    subtitle = Text("Q0.6B  •  CPU  •  OFFLINE  •  16 kHz MONO", style="dim", justify="center")
    header = Panel(Group(Align.center(title), Align.center(subtitle)), box=box.DOUBLE, border_style="bright_cyan", padding=(0, 1))

    status_table = Table.grid(expand=True, padding=(0, 1))
    status_table.add_column(ratio=1)
    status_table.add_column(ratio=1)
    status_table.add_row(
        Text("STATUS", style="bold dim"),
        Text(status_text, style=status_style),
    )
    status_table.add_row(
        Text("UTTERANCES", style="bold dim"),
        Text(str(snapshot["utterance_count"]), style="bold white"),
    )
    status_table.add_row(
        Text("IGNORED SEGMENTS", style="bold dim"),
        Text(str(snapshot["ignored_count"]), style="white"),
    )

    # Audio level meter. Scale is intentionally visual, not a calibrated dB meter.
    level = max(0.0, min(1.0, snapshot["mic_rms"] * 45.0))
    filled = int(level * 30)
    meter = Text("MIC  ", style="bold")
    meter.append("█" * filled, style="bright_green" if level < 0.75 else "bright_red")
    meter.append("░" * (30 - filled), style="dim")
    meter.append(f"  RMS {snapshot['mic_rms']:.4f}   PEAK {snapshot['mic_peak']:.4f}")

    noise = Text(
        f"NOISE FLOOR  {snapshot['noise_rms']:.5f}      "
        f"VAD MODE  {VAD_MODE}      "
        f"QUEUE  {snapshot['last_queue_seconds']:.1f}s",
        style="dim",
    )

    performance = Text(
        f"LAST DECODE  {snapshot['last_decode']:.2f}s      "
        f"RTF  {snapshot['last_rtf']:.2f}      "
        f"DROPPED AUDIO BLOCKS  {snapshot['dropped_blocks']}",
        style="dim",
    )

    status_body = Group(status_table, Text(""), meter, noise, performance)
    status_panel = Panel(status_body, title="[bold]MICROPHONE / ENGINE[/bold]", border_style="blue", box=box.ROUNDED)

    transcript_items = []
    if not snapshot["history"]:
        transcript_items.append(Text("Waiting for the first recognized sentence...", style="dim italic"))
    else:
        for stamp, original_text, accepted in snapshot["history"]:
            tag = "OK" if accepted else "CHECK"
            tag_style = "green" if accepted else "yellow"
            transcript_items.append(Text(f"{stamp}   [{tag}]", style=tag_style, justify="right"))
            # Reshape and reorder ONLY for display in terminals without RTL support.
            display_text = rtl_display(original_text)
            transcript_items.append(Text(display_text, style="bold white", justify="right", overflow="fold"))
            transcript_items.append(Text(""))

    transcript_panel = Panel(
        Group(*transcript_items),
        title="[bold]RECOGNIZED TEXT[/bold]",
        border_style="green",
        box=box.ROUNDED,
        padding=(1, 2),
    )

    event_text = Text(snapshot["last_event"], style="dim")
    if snapshot["error"]:
        event_text = Text(snapshot["error"], style="bold red", overflow="fold")

    footer_text = Text("Ctrl+C  Stop     •     Original Persian text is appended to transcriptions.txt", style="dim", justify="center")
    footer = Panel(Group(event_text, Align.center(footer_text)), border_style="dim", box=box.ROUNDED, padding=(0, 1))

    return Group(header, status_panel, transcript_panel, footer)


# ============================================================
# Main
# ============================================================
def main():
    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    LOG_FILE.touch(exist_ok=True)

    console.print(Panel.fit(
        "Loading Taraneh-Q0.6B on CPU...",
        title="[bold cyan]Starting[/bold cyan]",
        border_style="cyan",
    ))

    console.print(f"[dim]Model source:[/dim] {MODEL_ID}")
    console.print(f"[dim]Log file:[/dim] {LOG_FILE}")

    try:
        asr = Qwen3ASRModel.from_pretrained(
            MODEL_ID,
            dtype=torch.float32,
            device_map="cpu",
            max_inference_batch_size=1,
            max_new_tokens=128,
        )
    except Exception as exc:
        console.print(Panel(
            f"{type(exc).__name__}: {exc}\n\nCheck that the model files and Python packages are already available locally. Offline mode prevents automatic downloads.",
            title="[bold red]Model loading failed[/bold red]",
            border_style="red",
        ))
        raise

    info = sd.query_devices(INPUT_DEVICE, "input")
    if info["max_input_channels"] < 1:
        raise RuntimeError(f"Input device {INPUT_DEVICE} has no input channel")

    segmenter = UtteranceSegmenter()
    worker = threading.Thread(target=asr_worker, args=(asr,), daemon=True)
    worker.start()

    utterance_number = 0
    pending = np.empty(0, dtype=np.float32)

    console.clear()
    try:
        with sd.InputStream(
            device=INPUT_DEVICE,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=audio_callback,
            latency="high",
        ):
            update_state(last_event=f"Microphone ready: {info['name']}")

            with Live(
                build_dashboard(),
                console=console,
                refresh_per_second=8,
                screen=True,
                transient=False,
                redirect_stdout=False,
                redirect_stderr=False,
                vertical_overflow="crop",
            ) as live:
                while True:
                    # Drain PortAudio status messages.
                    while True:
                        try:
                            audio_status = status_queue.get_nowait()
                        except queue.Empty:
                            break
                        update_state(last_event=f"Audio status: {audio_status}")

                    with callback_drop_lock:
                        dropped = callback_drop_count
                    update_state(dropped_blocks=dropped)

                    try:
                        block = audio_queue.get(timeout=0.05)
                    except queue.Empty:
                        live.update(build_dashboard())
                        continue

                    pending = np.concatenate((pending, block))

                    while pending.size >= FRAME_SAMPLES:
                        frame = pending[:FRAME_SAMPLES].copy()
                        pending = pending[FRAME_SAMPLES:]

                        frame_rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
                        frame_peak = float(np.max(np.abs(frame)))

                        event, payload = segmenter.push(frame)

                        update_state(
                            mic_rms=frame_rms,
                            mic_peak=frame_peak,
                            noise_rms=segmenter.noise_rms,
                            speech_active=segmenter.active,
                        )

                        if event == "start":
                            update_state(last_event="Speech detected")

                        elif event == "end":
                            utterance_number += 1
                            frames, voiced_ratio = payload
                            try:
                                asr_queue.put_nowait((utterance_number, frames, voiced_ratio))
                                update_state(
                                    speech_active=False,
                                    last_event=f"Utterance {utterance_number} queued for recognition",
                                )
                            except queue.Full:
                                update_state(
                                    speech_active=False,
                                    ignored_count=ui_state["ignored_count"] + 1,
                                    last_event="ASR queue full; utterance skipped",
                                )

                        elif event == "discard":
                            update_state(
                                speech_active=False,
                                ignored_count=ui_state["ignored_count"] + 1,
                                last_event=f"Segment ignored: {payload}",
                            )

                    live.update(build_dashboard())

    except KeyboardInterrupt:
        pass
    finally:
        # Submit any unfinished utterance when stopping.
        event, payload = segmenter.flush()
        if event == "end":
            utterance_number += 1
            frames, voiced_ratio = payload
            try:
                asr_queue.put((utterance_number, frames, voiced_ratio), timeout=1.0)
            except queue.Full:
                pass

        # Stop worker after already queued utterances.
        try:
            asr_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        worker.join(timeout=120)

    console.print(Panel(
        f"Stopped.\nLog file: {LOG_FILE}",
        title="[bold green]Taraneh stopped[/bold green]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
