# Taraneh-Q0.6B — Quick Setup

Automated environment setup for the offline Persian live speech-recognition script.

## Requirements

- 64-bit Python 3.12
- Microphone
- Internet connection for the initial package/model download
- Disk space for PyTorch and the model

## Linux

From the project directory, copy `requirements.txt` and `install_linux.sh`, then run:

```bash
chmod +x install_linux.sh
./install_linux.sh
source .venv/bin/activate
python speech_recognition.py
```

The installer creates `.venv`, installs CPU-only PyTorch and Python dependencies,
and downloads the model to `models/Taraneh-Q0.6B`.

## Windows PowerShell

Run PowerShell in the project directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install_windows.ps1
.\.venv\Scripts\Activate.ps1
python speech_recognition.py
```

## Important

1. Put your application script (for example, `speech_recognition.py`) in this directory.
2. Set `INPUT_DEVICE` in the script to your microphone's device index.
3. The application runs in offline mode. Download the model before running it.
4. These installers use CPU-only PyTorch because the supplied script sets
   `device_map="cpu"` and `dtype=torch.float32`.
5. The model is downloaded from:
   https://huggingface.co/areffarhadi/Taraneh-Q0.6B

To list microphone devices:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```
