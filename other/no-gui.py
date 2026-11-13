
# ============================================================
# Offline mode
# Must be configured before importing Hugging Face libraries.
# ============================================================

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


# ============================================================
# Imports
# ============================================================

import time
import queue
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np
import sounddevice as sd
import torch
import webrtcvad

from qwen_asr import Qwen3ASRModel


# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Local model directory
LOCAL_MODEL_DIR = (
    SCRIPT_DIR / "models" / "Taraneh-Q0.6B"
)

# Use local model if available; otherwise use HF cache.
if (LOCAL_MODEL_DIR / "config.json").is_file():
    MODEL_ID = str(LOCAL_MODEL_DIR)
else:
    MODEL_ID = "areffarhadi/Taraneh-Q0.6B"


# ------------------------------------------------------------
# Microphone
# ------------------------------------------------------------

INPUT_DEVICE = 4

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1024


# ------------------------------------------------------------
# Output log
# ------------------------------------------------------------

LOG_FILE = SCRIPT_DIR / "transcriptions.txt"


# ------------------------------------------------------------
# Audio frames
# ------------------------------------------------------------

FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


# ============================================================
# Voice Activity Detection
# ============================================================

# WebRTC VAD:
# 0 = least aggressive
# 3 = most aggressive

VAD_MODE = 2

# No artificial amplification before VAD.
VAD_GAIN = 1.0

# Number of voiced frames required to start detection.
START_VOICED_FRAMES = 6

# Preserve audio before speech onset.
PRE_ROLL_MS = 350

# Silence required to finish an utterance.
END_SILENCE_MS = 700

# Preserve a little silence at the end.
KEEP_TAIL_MS = 220


# ============================================================
# Adaptive noise gate
# ============================================================

# Minimum RMS for a 20 ms audio frame.
MIN_FRAME_RMS = 0.0018

# Energy must exceed the estimated noise floor by this factor.
NOISE_MULTIPLIER = 1.8

# Initial noise estimate.
INITIAL_NOISE_RMS = 0.0008

# Noise estimation update speed.
NOISE_UPDATE_ALPHA = 0.02


# ============================================================
# Utterance quality filters
# ============================================================

MIN_VOICED_MS = 260

MIN_UTTERANCE_SECONDS = 0.60

MIN_VOICED_RATIO = 0.45

MAX_UTTERANCE_SECONDS = 20


# ============================================================
# Whole audio quality
# ============================================================

MIN_RMS = 0.0021

MIN_PEAK = 0.0085


# ============================================================
# Text validation
# ============================================================

MAX_WORDS_PER_SECOND = 8.0


# ============================================================
# CPU configuration
# ============================================================

CPU_THREADS = min(4, os.cpu_count() or 2)


# ============================================================
# Queues
# ============================================================

audio_queue = queue.Queue(maxsize=2000)

status_queue = queue.Queue(maxsize=32)

dropped_blocks = 0


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
    language="Persian"
):
    """
    Append every non-empty ASR result to a UTF-8 text file.

    Existing log entries are preserved.
    Timestamp uses local system timezone.
    """

    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    status = "ACCEPTED" if accepted else "REJECTED"

    entry = (
        f"[{timestamp}] "
        f"[Utterance={number}] "
        f"[{status}] "
        f"[Language={language}] "
        f"[Duration={duration:.2f}s] "
        f"[RMS={rms:.5f}] "
        f"[Peak={peak:.5f}] "
        f"[VAD={voiced_ratio:.2f}] "
        f"[Decode={decode_time:.2f}s]"
    )

    if reason:
        entry += f" [Reason={reason}]"

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
        newline="\n"
    ) as log:

        log.write(entry + "\n")
        log.write(text + "\n")
        log.write("-" * 80 + "\n")
        log.flush()


# ============================================================
# Audio callback
# ============================================================

def audio_callback(indata, frames, time_info, status):

    global dropped_blocks

    if status:

        try:
            status_queue.put_nowait(str(status))

        except queue.Full:
            pass

    try:

        audio_queue.put_nowait(
            indata[:, 0].copy()
        )

    except queue.Full:

        dropped_blocks += 1


# ============================================================
# VAD + Adaptive Noise Gate
# ============================================================

class UtteranceSegmenter:

    def __init__(self):

        self.vad = webrtcvad.Vad(VAD_MODE)

        self.pre_roll_count = (
            PRE_ROLL_MS // FRAME_MS
        )

        self.start_count = START_VOICED_FRAMES

        self.end_silence_count = (
            END_SILENCE_MS // FRAME_MS
        )

        self.keep_tail_count = (
            KEEP_TAIL_MS // FRAME_MS
        )

        self.min_voiced_count = (
            MIN_VOICED_MS // FRAME_MS
        )

        self.max_frames = int(
            MAX_UTTERANCE_SECONDS * 1000 / FRAME_MS
        )

        self.pre_roll = deque(
            maxlen=self.pre_roll_count
        )

        self.noise_rms = INITIAL_NOISE_RMS

        self.reset()

    # --------------------------------------------------------
    # Reset current utterance state
    # --------------------------------------------------------

    def reset(self):

        self.active = False

        self.frames = []

        self.voiced_count = 0

        self.silence_count = 0

    # --------------------------------------------------------
    # Detect speech in one frame
    # --------------------------------------------------------

    def is_voiced(self, frame):

        frame_rms = float(
            np.sqrt(np.mean(frame ** 2))
        )

        vad_frame = np.clip(
            frame * VAD_GAIN,
            -1.0,
            1.0
        )

        pcm16 = (
            vad_frame * 32767.0
        ).astype("<i2")

        vad_result = self.vad.is_speech(
            pcm16.tobytes(),
            SAMPLE_RATE
        )

        # Update noise estimation only when idle
        # and WebRTC VAD does not detect speech.
        if not self.active and not vad_result:

            self.noise_rms = (
                (1.0 - NOISE_UPDATE_ALPHA) * self.noise_rms
                +
                NOISE_UPDATE_ALPHA * frame_rms
            )

        # Adaptive threshold
        energy_threshold = max(
            MIN_FRAME_RMS,
            self.noise_rms * NOISE_MULTIPLIER
        )

        energy_ok = (
            frame_rms >= energy_threshold
        )

        return vad_result and energy_ok

    # --------------------------------------------------------
    # Finalize an utterance
    # --------------------------------------------------------

    def finalize(self):

        if not self.frames:

            self.reset()
            self.pre_roll.clear()

            return "discard", "empty segment"

        # Remove excess trailing silence.
        trim = max(
            0,
            self.silence_count - self.keep_tail_count
        )

        if trim:

            final_frames = self.frames[:-trim]

        else:

            final_frames = self.frames[:]

        duration = (
            len(final_frames) * FRAME_MS / 1000.0
        )

        voiced_ms = (
            self.voiced_count * FRAME_MS
        )

        voiced_ratio = (
            self.voiced_count /
            max(len(final_frames), 1)
        )

        # Reset segment state.
        self.reset()
        self.pre_roll.clear()

        if duration < MIN_UTTERANCE_SECONDS:

            return "discard", (
                f"too short: {duration:.2f}s"
            )

        if voiced_ms < MIN_VOICED_MS:

            return "discard", (
                f"insufficient voiced audio: {voiced_ms}ms"
            )

        if voiced_ratio < MIN_VOICED_RATIO:

            return "discard", (
                f"low voiced ratio: {voiced_ratio:.2f}"
            )

        return "end", (
            final_frames,
            voiced_ratio
        )

    # --------------------------------------------------------
    # Process one 20 ms frame
    # --------------------------------------------------------

    def push(self, frame):

        voiced = self.is_voiced(frame)

        # Waiting for speech.
        if not self.active:

            self.pre_roll.append(
                (frame, voiced)
            )

            recent_voiced = sum(
                1
                for _, flag in self.pre_roll
                if flag
            )

            if recent_voiced >= self.start_count:

                self.active = True

                self.frames = [
                    samples
                    for samples, _ in self.pre_roll
                ]

                self.voiced_count = recent_voiced

                self.silence_count = 0

                return "start", None

            return None, None

        # Active speech recording.
        self.frames.append(frame)

        if voiced:

            self.voiced_count += 1

            self.silence_count = 0

        else:

            self.silence_count += 1

        silence_done = (
            self.silence_count >= self.end_silence_count
        )

        length_done = (
            len(self.frames) >= self.max_frames
        )

        if silence_done or length_done:

            return self.finalize()

        return None, None

    # --------------------------------------------------------
    # Flush remaining audio on Ctrl+C
    # --------------------------------------------------------

    def flush(self):

        if not self.active:

            return None, None

        return self.finalize()


# ============================================================
# Text validation
# ============================================================

def validate_text(text, duration):

    words = text.strip().split()

    if not words:

        return False, "empty transcription"

    word_rate = (
        len(words) / max(duration, 0.01)
    )

    if word_rate > MAX_WORDS_PER_SECOND:

        return False, (
            f"unrealistic word rate: {word_rate:.1f}/s"
        )

    # Detect repeated sequences of 3 or 4 words.
    for n in (3, 4):

        for i in range(
            len(words) - 2 * n + 1
        ):

            first = words[i:i+n]

            second = words[i+n:i+2*n]

            if first == second:

                return False, "repeated phrase detected"

    return True, "OK"


# ============================================================
# Qwen ASR transcription
# ============================================================

def transcribe_utterance(
    asr,
    frames,
    number,
    voiced_ratio
):

    audio = np.concatenate(
        frames
    ).astype(np.float32)

    duration = len(audio) / SAMPLE_RATE

    # Remove DC offset.
    audio = audio - np.mean(audio)

    peak = float(
        np.max(np.abs(audio))
    )

    rms = float(
        np.sqrt(np.mean(audio ** 2))
    )

    print(
        f"\n[Utterance {number}: "
        f"{duration:.2f}s | "
        f"RMS={rms:.4f} | "
        f"Peak={peak:.4f} | "
        f"VAD={voiced_ratio:.2f}]",
        flush=True
    )

    # Reject low-quality audio before ASR.
    if rms < MIN_RMS:

        print(
            "[Rejected: low audio energy]",
            flush=True
        )

        return

    if peak < MIN_PEAK:

        print(
            "[Rejected: low peak level]",
            flush=True
        )

        return

    started = time.perf_counter()

    try:

        results = asr.transcribe(
            audio=(audio, SAMPLE_RATE),
            language="Persian",
        )

        decode_time = (
            time.perf_counter() - started
        )

        if not results:

            print(
                "[No ASR result]",
                flush=True
            )

            return

        result = results[0]

        text = str(
            getattr(result, "text", "") or ""
        ).strip()

        detected_language = getattr(
            result,
            "language",
            "Persian"
        )

        if not text:

            print(
                "[Empty transcription]",
                flush=True
            )

            return

        valid, reason = validate_text(
            text,
            duration
        )

        # Save every non-empty ASR output.
        # Rejected text is also saved.
        append_transcription_log(
            text=text,
            number=number,
            duration=duration,
            rms=rms,
            peak=peak,
            voiced_ratio=voiced_ratio,
            accepted=valid,
            reason="" if valid else reason,
            decode_time=decode_time,
            language=detected_language,
        )

        if valid:

            print(
                f"Recognized: {text}",
                flush=True
            )

        else:

            print(
                f"[Rejected transcription: {reason}]",
                flush=True
            )

            print(
                f"Raw ASR text saved: {text}",
                flush=True
            )

        rtf = (
            decode_time / max(duration, 0.001)
        )

        backlog = (
            audio_queue.qsize() *
            BLOCK_SIZE / SAMPLE_RATE
        )

        print(
            f"[Language={detected_language} | "
            f"Decode={decode_time:.2f}s | "
            f"RTF={rtf:.2f} | "
            f"Queued={backlog:.1f}s]",
            flush=True
        )

    except Exception as exc:

        print(
            f"[ASR ERROR] "
            f"{type(exc).__name__}: {exc}",
            flush=True
        )


# ============================================================
# Main
# ============================================================

def main():

    global dropped_blocks

    torch.set_num_threads(CPU_THREADS)

    try:

        torch.set_num_interop_threads(1)

    except RuntimeError:

        pass

    # Check local model files.
    if Path(MODEL_ID).is_dir():

        if not (
            Path(MODEL_ID) / "config.json"
        ).is_file():

            raise FileNotFoundError(
                f"Invalid model directory: {MODEL_ID}"
            )

    # Create log file if needed.
    LOG_FILE.touch(exist_ok=True)

    print("Loading Taraneh-Q0.6B on CPU...")

    print("Model path:", MODEL_ID)

    asr = Qwen3ASRModel.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        device_map="cpu",
        max_inference_batch_size=1,
        max_new_tokens=128,
    )

    print("Model loaded.")

    # Check microphone.
    info = sd.query_devices(
        INPUT_DEVICE,
        "input"
    )

    if info["max_input_channels"] < 1:

        raise RuntimeError(
            "Selected microphone has no input channel"
        )

    print(
        f"Microphone: [{INPUT_DEVICE}] {info['name']}"
    )

    print(
        f"Log file: {LOG_FILE}"
    )

    segmenter = UtteranceSegmenter()

    utterance_number = 0

    pending = np.empty(
        0,
        dtype=np.float32
    )

    last_reported_drops = 0

    print("\nConfiguration:")
    print("VAD mode:", VAD_MODE)
    print("Frame RMS threshold:", MIN_FRAME_RMS)
    print("Noise multiplier:", NOISE_MULTIPLIER)
    print("Minimum utterance:", MIN_UTTERANCE_SECONDS)
    print("Minimum voiced ratio:", MIN_VOICED_RATIO)

    print("\nOpening microphone...")

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

            print("Microphone opened.")

            print(
                "\nListening. Speak Persian and pause briefly."
            )

            print("Press Ctrl+C to stop.\n")

            while True:

                # Report audio-device warnings.
                while True:

                    try:

                        status = status_queue.get_nowait()

                    except queue.Empty:

                        break

                    print(
                        f"[Audio status: {status}]",
                        flush=True
                    )

                # Report dropped audio blocks.
                if dropped_blocks > last_reported_drops:

                    lost = (
                        dropped_blocks - last_reported_drops
                    )

                    last_reported_drops = dropped_blocks

                    print(
                        f"[WARNING: dropped {lost} audio blocks]",
                        flush=True
                    )

                try:

                    block = audio_queue.get(
                        timeout=0.1
                    )

                except queue.Empty:

                    continue

                pending = np.concatenate(
                    (pending, block)
                )

                # Process exact 20 ms frames.
                while pending.size >= FRAME_SAMPLES:

                    frame = pending[
                        :FRAME_SAMPLES
                    ].copy()

                    pending = pending[
                        FRAME_SAMPLES:
                    ]

                    event, payload = segmenter.push(
                        frame
                    )

                    if event == "start":

                        print(
                            "\n[Speech detected]",
                            flush=True
                        )

                    elif event == "end":

                        utterance_number += 1

                        frames, voiced_ratio = payload

                        transcribe_utterance(
                            asr,
                            frames,
                            utterance_number,
                            voiced_ratio
                        )

                        print(
                            "[Listening...]",
                            flush=True
                        )

                    elif event == "discard":

                        print(
                            f"[Segment ignored: {payload}]",
                            flush=True
                        )

    except KeyboardInterrupt:

        print("\nStopping...")

        event, payload = segmenter.flush()

        if event == "end":

            utterance_number += 1

            frames, voiced_ratio = payload

            transcribe_utterance(
                asr,
                frames,
                utterance_number,
                voiced_ratio
            )

        elif event == "discard":

            print(
                f"[Final segment ignored: {payload}]",
                flush=True
            )

        print("\nStreaming stopped.")

        print(
            f"Transcript log: {LOG_FILE}"
        )


if __name__ == "__main__":

    main()
