import bisect
from datetime import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from core.file_manager import PodcastFileManager

FILE_MANAGER = PodcastFileManager()

import av
import numpy as np
import pyqtgraph as pg
import shiboken6
import torch

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QAction, QColor, QCursor, QFont, QIcon, QKeyEvent, QKeySequence,
    QShortcut, QTextBlockFormat, QTextCharFormat, QTextCursor,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox,
    QPushButton, QScrollBar, QSlider, QSplitter, QTextEdit, QVBoxLayout, QWidget,
)

# ── Configuration & Paths ──────────────────────────────────────────────────────

WAVEFORM_CACHE_DIR = FILE_MANAGER.base_dir / "waveform_cache"

MODELS = [
    {"name": "Large v3 Turbo（推薦）", "repo": "mlx-community/whisper-large-v3-turbo",  "size_mb": 809,  "rtf": 0.075},
    {"name": "Large v3",               "repo": "mlx-community/whisper-large-v3-mlx",    "size_mb": 3000, "rtf": 0.15},
    {"name": "Large v2",               "repo": "mlx-community/whisper-large-v2-mlx",    "size_mb": 3000, "rtf": 0.15},
    {"name": "Medium",                 "repo": "mlx-community/whisper-medium-mlx",       "size_mb": 1500, "rtf": 0.08},
    {"name": "Small",                  "repo": "mlx-community/whisper-small-mlx",        "size_mb": 500,  "rtf": 0.05},
]

_HL_BG  = QColor("#2563EB")
_HL_FG  = QColor("white")
_DEL_BG = QColor(220, 38, 38, 160)
_DEL_FG = QColor("white")
_VAD_BG = QColor(234, 179, 8, 160)
_VAD_FG = QColor("#1a1a1a")
_FILLER_BG       = QColor("#f59e0b")
_FILLER_FG       = QColor("white")
_SEARCH_ACTIVE_BG = QColor("#047857")
_SEARCH_ACTIVE_FG = QColor("white")
_SEARCH_BG       = QColor("#10B981")
_SEARCH_FG       = QColor("white")

pg.setConfigOptions(antialias=True)


def qt_obj_alive(obj) -> bool:
    return obj is not None and shiboken6.isValid(obj)


def fmt_time(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def compute_keep_ranges(
    duration_s: float, delete_ranges: list[dict]
) -> list[tuple[float, float]]:
    deletes = sorted(delete_ranges, key=lambda r: r["start"])
    keep, cursor = [], 0.0
    for d in deletes:
        d_start = max(0.0, d["start"])
        d_end   = min(duration_s, d["end"])
        if d_start > cursor:
            keep.append((cursor, d_start))
        cursor = max(cursor, d_end)
    if cursor < duration_s:
        keep.append((cursor, duration_s))
    return keep


def yield_resampled_mono_chunks(path: str, target_sr: int):
    with av.open(path) as container:
        audio = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                yield out.to_ndarray()[0].astype(np.float32, copy=False)
        for out in resampler.resample(None):
            yield out.to_ndarray()[0].astype(np.float32, copy=False)


def merge_time_segments(
    segments: list[dict], max_gap_s: float = 0.0, min_duration_s: float = 0.0
) -> list[dict]:
    if not segments:
        return []
    merged: list[dict] = []
    for seg in sorted(segments, key=lambda s: (s["start"], s["end"])):
        start = float(seg["start"])
        end   = float(seg["end"])
        if end <= start:
            continue
        if not merged:
            merged.append({"start": start, "end": end})
            continue
        prev = merged[-1]
        if start <= prev["end"] + max_gap_s:
            prev["end"] = max(prev["end"], end)
        else:
            merged.append({"start": start, "end": end})
    if min_duration_s <= 0:
        return merged
    return [s for s in merged if s["end"] - s["start"] >= min_duration_s]


def transcript_path_for(audio_path: str) -> Path:
    project_dir = FILE_MANAGER.get_project_dir(audio_path)
    return project_dir / "transcript.json"


def is_model_cached(repo_id: str) -> bool:
    cache_dir  = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir   = "models--" + repo_id.replace("/", "--")
    snapshots  = cache_dir / repo_dir / "snapshots"
    return snapshots.exists() and any(snapshots.iterdir())


def audio_duration_s(path: str) -> float:
    try:
        with av.open(path) as c:
            return float(c.duration) / 1_000_000 if c.duration else 0.0
    except Exception:
        return 0.0


def _worker_is_cancelled() -> bool:
    t = QThread.currentThread()
    return t is not None and t.isInterruptionRequested()




# ── Lufs worker ───────────────────────────────────────────────────────────────

class LufsWorker(QObject):
    finished = Signal(str)

    def __init__(self, audio_path: str):
        super().__init__()
        self._audio_path = audio_path

    def run(self) -> None:
        try:
            cmd = ["ffmpeg", "-nostats", "-i", self._audio_path, "-af", "ebur128=framelog=verbose", "-f", "null", "-"]
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            output = proc.stderr
            lufs_val = ""
            lines = output.splitlines()
            for i, line in enumerate(lines):
                if "Integrated loudness:" in line:
                    if i + 1 < len(lines):
                        parts = lines[i+1].strip().split()
                        if len(parts) >= 2 and parts[0] == "I:":
                            lufs_val = parts[1]
                            break
            if lufs_val:
                self.finished.emit(f"LUFS: {lufs_val}")
            else:
                self.finished.emit("LUFS: 未知")
        except Exception as e:
            self.finished.emit("LUFS: 錯誤")


# ── Waveform worker ────────────────────────────────────────────────────────────

class WaveformWorker(QObject):
    finished = Signal(int, object, float)

    def __init__(self, job_id: int, audio_path: str, cache_dir: Path, bucket_ms: int = 1):
        super().__init__()
        self._job_id     = job_id
        self._path       = audio_path
        self._cache_dir  = cache_dir
        self._bucket_ms  = bucket_ms

    def _cache_path(self) -> Path:
        src  = Path(self._path).resolve()
        stat = src.stat()
        key  = "|".join([str(src), str(stat.st_size),
                          str(int(stat.st_mtime_ns)), str(self._bucket_ms)])
        return self._cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()}.npz"

    def run(self) -> None:
        try:
            cache_path = self._cache_path()
            if cache_path.exists():
                with np.load(cache_path) as c:
                    self.finished.emit(self._job_id,
                                       c["peaks"].astype(np.float32, copy=False),
                                       float(c["duration_s"]))
                return

            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with av.open(self._path) as container:
                audio       = container.streams.audio[0]
                sr          = audio.codec_context.sample_rate
                resampler   = av.AudioResampler(format="fltp", layout="mono", rate=sr)
                bucket_size = max(1, int(sr * self._bucket_ms / 1000.0))
                carry       = np.empty(0, dtype=np.float32)
                parts: list[np.ndarray] = []
                total = 0

                def _process(mono):
                    nonlocal carry, total
                    total += len(mono)
                    if carry.size:
                        mono  = np.concatenate((carry, mono))
                        carry = np.empty(0, dtype=np.float32)
                    n = len(mono) // bucket_size
                    if n:
                        parts.append(
                            np.abs(mono[:n*bucket_size].reshape(n, bucket_size))
                            .max(axis=1).astype(np.float32, copy=False))
                    rem = mono[n*bucket_size:]
                    if rem.size:
                        carry = rem.copy()

                for frame in container.decode(audio=0):
                    if _worker_is_cancelled(): return
                    for out in resampler.resample(frame):
                        _process(out.to_ndarray()[0].astype(np.float32, copy=False))
                for out in resampler.resample(None):
                    if _worker_is_cancelled(): return
                    _process(out.to_ndarray()[0].astype(np.float32, copy=False))

            if carry.size:
                parts.append(np.array([float(np.abs(carry).max())], dtype=np.float32))
            peaks = np.concatenate(parts).astype(np.float32) if parts else np.zeros(1, np.float32)
            dur   = total / sr if sr else 0.0
            if not _worker_is_cancelled():
                np.savez_compressed(cache_path, peaks=peaks, duration_s=np.float32(dur))
                self.finished.emit(self._job_id, peaks, dur)
        except Exception as exc:
            print(f"[WaveformWorker] {exc}", flush=True)
            self.finished.emit(self._job_id, np.zeros(1, np.float32), 0.0)


# ── VAD worker ─────────────────────────────────────────────────────────────────

class VadWorker(QObject):
    finished = Signal(int, list)
    status   = Signal(int, str)
    error    = Signal(int, str)

    def __init__(self, job_id: int, audio_path: str, duration_s: float,
                 threshold: float = 0.5, min_silence_ms: int = 300, min_speech_ms: int = 100):
        super().__init__()
        self._job_id     = job_id
        self._path       = audio_path
        self._duration_s = duration_s
        self._threshold  = threshold
        self._min_silence = min_silence_ms
        self._min_speech  = min_speech_ms

    def run(self) -> None:
        try:
            from silero_vad import VADIterator, load_silero_vad
            self.status.emit(self._job_id, "正在載入 VAD 模型…")
            model    = load_silero_vad()
            iterator = VADIterator(model, threshold=self._threshold,
                                   sampling_rate=16000,
                                   min_silence_duration_ms=self._min_silence)
            carry          = np.empty(0, dtype=np.float32)
            current_start: float | None = None
            speech: list[dict] = []
            total = 0
            window = 512
            n_win  = 0
            stride = max(1, int(16000 * 10 / window))

            self.status.emit(self._job_id, "執行串流 VAD…")
            for mono in yield_resampled_mono_chunks(self._path, 16000):
                if _worker_is_cancelled(): return
                total += len(mono)
                if carry.size:
                    mono  = np.concatenate((carry, mono))
                    carry = np.empty(0, dtype=np.float32)
                n_full = len(mono) // window
                for i in range(n_full):
                    ev = iterator(torch.from_numpy(mono[i*window:(i+1)*window]), return_seconds=True)
                    if ev:
                        if "start" in ev:   current_start = float(ev["start"])
                        elif "end" in ev and current_start is not None:
                            speech.append({"start": current_start, "end": float(ev["end"])})
                            current_start = None
                    n_win += 1
                    if n_win % stride == 0:
                        self.status.emit(self._job_id, f"執行串流 VAD… {n_win*window/16000:.0f}s")
                rem = mono[n_full*window:]
                if rem.size: carry = rem.copy()

            if carry.size and not _worker_is_cancelled():
                padded = np.pad(carry, (0, window - len(carry))).astype(np.float32)
                ev = iterator(torch.from_numpy(padded), return_seconds=True)
                if ev:
                    if "start" in ev: current_start = float(ev["start"])
                    elif "end" in ev and current_start is not None:
                        speech.append({"start": current_start, "end": float(ev["end"])})
                        current_start = None

            dur = max(self._duration_s, total / 16000.0)
            if current_start is not None:
                speech.append({"start": current_start, "end": dur})

            speech_ts = merge_time_segments(speech,
                                            max_gap_s=max(self._min_silence / 1000, 0.05),
                                            min_duration_s=self._min_speech / 1000)
            silence: list[dict] = []
            prev = 0.0
            for seg in speech_ts:
                if seg["start"] > prev + 0.05:
                    silence.append({"start": round(prev, 3), "end": round(seg["start"], 3)})
                prev = seg["end"]
            if prev < dur - 0.05:
                silence.append({"start": round(prev, 3), "end": round(dur, 3)})

            if not _worker_is_cancelled():
                self.finished.emit(self._job_id, silence)
        except Exception as exc:
            import traceback
            self.error.emit(self._job_id, f"VAD 錯誤：{exc}\n{traceback.format_exc()}")


# ── Export worker ──────────────────────────────────────────────────────────────

class ExportWorker(QObject):
    finished = Signal(str)
    error    = Signal(str)
    progress = Signal(str)
    elapsed  = Signal(float)

    def __init__(self, audio_path: str, keep_ranges: list[tuple[float, float]], output_path: str):
        super().__init__()
        self._audio_path  = audio_path
        self._keep_ranges = keep_ranges
        self._output_path = output_path

    @staticmethod
    def _audio_codec_args(in_path: str, out_path: str) -> list[str]:
        in_ext  = Path(in_path).suffix.lower()
        out_ext = Path(out_path).suffix.lower()
        if in_ext == out_ext and in_ext in (".m4a", ".mp4", ".aac"):
            return ["-c", "copy"]
        if out_ext in (".m4a", ".mp4", ".aac"):
            return ["-c:a", "aac", "-b:a", "192k"]
        if out_ext == ".wav":
            return ["-c:a", "pcm_s16le"]
        return ["-c", "copy"]

    def run(self) -> None:
        try:
            import time
            t0 = time.time()
            total = len(self._keep_ranges)
            out_ext = Path(self._output_path).suffix.lower()
            seg_ext = out_ext if out_ext in (".m4a", ".mp4", ".wav") else ".m4a"
            codec   = self._audio_codec_args(self._audio_path, self._output_path)
            with tempfile.TemporaryDirectory() as tmpdir:
                segs: list[str] = []
                for i, (s, e) in enumerate(self._keep_ranges):
                    self.progress.emit(f"匯出中… 切段 {i+1}/{total}（{s:.1f}s – {e:.1f}s）")
                    seg = os.path.join(tmpdir, f"seg_{i:03d}{seg_ext}")
                    err_path = os.path.join(tmpdir, f"err_seg_{i}.log")
                    with open(err_path, "w") as err_f:
                        r = subprocess.run(
                            ["ffmpeg", "-y", "-i", self._audio_path,
                             "-ss", str(s), "-to", str(e)] + codec + [seg],
                            stdout=subprocess.DEVNULL, stderr=err_f)
                    if r.returncode != 0:
                        with open(err_path, "r") as err_f:
                            err_str = err_f.read()[-800:]
                        self.error.emit(f"FFmpeg 分段錯誤:\n{err_str}"); return
                    segs.append(seg)
                self.progress.emit(f"匯出中… 合併 {total} 個片段…")
                concat = os.path.join(tmpdir, "concat.txt")
                with open(concat, "w") as f:
                    f.writelines(f"file '{s}'\n" for s in segs)
                merge_codec = ["-c", "copy"] if seg_ext == out_ext else codec
                err_path = os.path.join(tmpdir, "err_merge.log")
                with open(err_path, "w") as err_f:
                    r = subprocess.run(
                        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                         "-i", concat] + merge_codec + [self._output_path],
                        stdout=subprocess.DEVNULL, stderr=err_f)
                if r.returncode != 0:
                    with open(err_path, "r") as err_f:
                        err_str = err_f.read()[-800:]
                    self.error.emit(f"FFmpeg 合併錯誤:\n{err_str}"); return
            elapsed = time.time() - t0
            self.elapsed.emit(elapsed)
            self.finished.emit(self._output_path)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Transcribe worker (mlx-whisper) ───────────────────────────────────────────

class TranscribeWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(int, dict)
    error    = Signal(int, str)

    def __init__(self, job_id: int, audio_path: str, repo_id: str, output_path: str,
                 word_timestamps: bool = False, initial_prompt: str = ""):
        super().__init__()
        self._job_id          = job_id
        self._path            = audio_path
        self._repo_id         = repo_id
        self._output_path     = output_path
        self._word_timestamps = word_timestamps
        self._initial_prompt  = initial_prompt

    def run(self) -> None:
        try:
            import mlx_whisper
            cached = is_model_cached(self._repo_id)
            self.progress.emit(self._job_id,
                               "正在轉錄音檔，請稍候…" if cached else "正在下載並載入模型（首次需要較長時間）…")

            import sys, re, io
            class ProgressStream(io.StringIO):
                def __init__(self, signal, jid):
                    super().__init__()
                    self.signal = signal
                    self.jid = jid
                    self.last_val = -1
                def write(self, s):
                    sys.__stderr__.write(s)
                    m = re.search(r"(\d+)%", s)
                    if m:
                        val = int(m.group(1))
                        if val != self.last_val:
                            self.last_val = val
                            self.signal.emit(self.jid, f"正在轉錄音檔… {val}%")
                            
            old_stderr = sys.stderr
            sys.stderr = ProgressStream(self.progress, self._job_id)
            try:
                result = mlx_whisper.transcribe(
                    self._path,
                    path_or_hf_repo=self._repo_id,
                    word_timestamps=self._word_timestamps,
                    language="zh",
                    verbose=False,
                    initial_prompt=self._initial_prompt or None,
                )
            finally:
                sys.stderr = old_stderr

            self.progress.emit(self._job_id, "正在整理逐字稿…")
            tokens: list[dict] = []
            tid = 0
            dur = 0.0
            for seg in result.get("segments", []):
                dur = max(dur, float(seg.get("end", 0)))
                if self._word_timestamps:
                    for w in seg.get("words", []):
                        if w.get("start") is None or w.get("end") is None:
                            continue
                        tokens.append({
                            "id":         tid,
                            "text":       w.get("word", ""),
                            "start":      round(float(w["start"]), 3),
                            "end":        round(float(w["end"]),   3),
                            "confidence": round(float(w.get("probability", 1.0)), 4),
                            "segment_id": seg.get("id", 0),
                        })
                        tid += 1
                else:
                    # If not word timestamps, just treat segment as one token?
                    # Or maybe just don't have tokens for now?
                    # The UI seems to rely on tokens.
                    # Let's map segment to token.
                    tokens.append({
                        "id":         tid,
                        "text":       seg.get("text", ""),
                        "start":      round(float(seg.get("start", 0)), 3),
                        "end":        round(float(seg.get("end", 0)),   3),
                        "confidence": 1.0,
                        "segment_id": seg.get("id", 0),
                    })
                    tid += 1

            data = {
                "audio_path": str(Path(self._path).resolve()),
                "duration":   dur,
                "model":      self._repo_id,
                "segments": [
                    {"id": seg.get("id", i), "text": seg.get("text", ""),
                     "start": float(seg.get("start", 0)), "end": float(seg.get("end", 0))}
                    for i, seg in enumerate(result.get("segments", []))
                ],
                "tokens": tokens,
            }
            Path(self._output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self._output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.finished.emit(self._job_id, data)
        except Exception as exc:
            import traceback
            self.error.emit(self._job_id, f"轉錄失敗：{exc}\n{traceback.format_exc()}")


# ── Model select dialog ────────────────────────────────────────────────────────

class ModelSelectDialog(QDialog):
    def __init__(self, current_repo: str, audio_dur_s: float = 0.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("選擇轉錄模型與精細度")
        self.setMinimumWidth(440)
        self._repo = current_repo
        self._audio_dur_s = audio_dur_s

        layout = QVBoxLayout(self)
        
        # Precision selector
        layout.addWidget(QLabel("時間精細度："))
        self._prec_combo = QComboBox()
        self._prec_combo.addItem("段落級 (segment) — 最快", "segment")
        self._prec_combo.addItem("字詞級 (word) — 最精準（較慢）", "word")
        self._prec_combo.setCurrentIndex(0)
        self._prec_combo.currentIndexChanged.connect(self._update_list)
        layout.addWidget(self._prec_combo)

        layout.addWidget(QLabel("提示詞（節目名稱、人名等特定詞彙，提升辨識率）："))
        self._prompt_edit = QLineEdit()
        self._prompt_edit.setPlaceholderText("例：消業障旅行團、Brian、XX 頻道…")
        self._prompt_edit.setText(FILE_MANAGER.load_config().get("initial_prompt", ""))
        layout.addWidget(self._prompt_edit)

        layout.addWidget(QLabel("選擇 Whisper 模型："))
        self._list = QListWidget()
        self._list.currentItemChanged.connect(
            lambda item, _: setattr(self, "_repo", item.data(Qt.ItemDataRole.UserRole)) if item else None)
        layout.addWidget(self._list)

        self._update_list()

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _update_list(self):
        self._list.clear()
        mult = 2.0 if self.selected_precision() == "word" else 1.0
        for m in MODELS:
            cached = is_model_cached(m["repo"])
            est = ""
            if self._audio_dur_s > 0:
                secs = self._audio_dur_s * m["rtf"] * mult
                est  = f"   ⏱ 預計 {int(secs//60)} 分 {int(secs%60):02d} 秒" if secs >= 60 else f"   ⏱ 預計 {int(secs)} 秒"
            status = "✅ 已下載" if cached else f"⬇ 需下載 ~{m['size_mb']} MB"
            item = QListWidgetItem(f"{m['name']}\n{status}{est}")
            item.setData(Qt.ItemDataRole.UserRole, m["repo"])
            self._list.addItem(item)
            if m["repo"] == self._repo:
                self._list.setCurrentItem(item)

    def selected_repo(self) -> str:
        return self._repo

    def selected_precision(self) -> str:
        return self._prec_combo.currentData()

    def selected_prompt(self) -> str:
        return self._prompt_edit.text().strip()


# ── Delete region item ─────────────────────────────────────────────────────────

class DeleteRegionItem(pg.LinearRegionItem):
    sig_right_click = Signal(QPoint)

    def __init__(self, start_s: float, end_s: float):
        super().__init__(values=[start_s, end_s],
                         brush=pg.mkBrush(220, 38, 38, 90),
                         pen=pg.mkPen(None), movable=False)
        self.setZValue(10)
        self.setAcceptedMouseButtons(Qt.MouseButton.RightButton)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton:
            self.sig_right_click.emit(ev.screenPos().toPoint()); ev.accept()
        else:
            super().mouseClickEvent(ev)


# ── VAD region item ────────────────────────────────────────────────────────────

class VadRegionItem(pg.LinearRegionItem):
    sig_right_click = Signal(QPoint)

    def __init__(self, start_s: float, end_s: float, duration_s: float = 0.0):
        super().__init__(values=[start_s, end_s],
                         brush=pg.mkBrush(234, 179, 8, 70),
                         pen=pg.mkPen(234, 179, 8, 180), movable=False)
        self.setZValue(8)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)
        for line in self.lines:
            line.setMovable(True)
            line.setPen(pg.mkPen(234, 179, 8, 220, width=4))
            line.setHoverPen(pg.mkPen(234, 179, 8, 255, width=6))
        if duration_s > 0:
            self.setBounds([0.0, duration_s])

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton:
            self.sig_right_click.emit(ev.screenPos().toPoint()); ev.accept()
        else:
            # Pass other buttons (like LeftButton for panning) to the view
            ev.ignore()


# ── Waveform widget ────────────────────────────────────────────────────────────

class TimeAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        strings = []
        for v in values:
            s = int(v)
            ms_val = int((v - s) * 100)
            if spacing < 1.0:
                if s >= 3600:
                    strings.append(f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}.{ms_val:02d}")
                else:
                    strings.append(f"{s//60:02d}:{s%60:02d}.{ms_val:02d}")
            else:
                if s >= 3600:
                    strings.append(f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}")
                else:
                    strings.append(f"{s//60:02d}:{s%60:02d}")
        return strings

class WaveformWidget(pg.PlotWidget):
    seek_requested         = Signal(float)
    range_changed          = Signal(float, float)
    waveform_right_clicked = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent, axisItems={'bottom': TimeAxis(orientation='bottom')})
        self._pan_scene_x = None
        self._pan_view_x = None
        self._left_press_x = None
        self._left_dragging = False
        self._duration_s = 0.0
        self._vad_regions = {}
        self._del_regions = {}
        self.setMinimumHeight(60)
        self.setBackground("#111827")
        self.setMenuEnabled(False)
        self.hideAxis("left")
        self.plotItem.getViewBox().setMouseEnabled(x=True, y=False)
        self.setFocusPolicy(Qt.StrongFocus)

        self._bottom_axis = self.getAxis("bottom")
        self._bottom_axis.show()
        self._bottom_axis.setStyle(tickTextOffset=5, tickLength=-5)
        self._bottom_axis.setPen(pg.mkPen("#4a5568", width=1))
        self._bottom_axis.setTextPen(pg.mkPen("#9ca3af", width=1))

        wave_pen = pg.mkPen(color="#60a5fa", width=1)
        self._upper = self.plot(pen=wave_pen)
        self._lower = self.plot(pen=wave_pen)
        self._upper.setClipToView(True); self._lower.setClipToView(True)
        self._upper.setDownsampling(auto=True, method="peak")
        self._lower.setDownsampling(auto=True, method="peak")
        self._fill = pg.FillBetweenItem(self._upper, self._lower,
                                        brush=pg.mkBrush(96, 165, 250, 50))
        self.addItem(self._fill)

        self._playhead = pg.InfiniteLine(pos=0, angle=90,
                                         pen=pg.mkPen("#ef4444", width=2))
        self.addItem(self._playhead)

        self.plotItem.getViewBox().setMouseEnabled(x=True, y=False)
        self.scene().sigMouseClicked.connect(self._on_scene_clicked)
        self.plotItem.vb.sigRangeChanged.connect(
            lambda vb, r: self.range_changed.emit(*r[0]))

    def focusInEvent(self, ev):
        self.setStyleSheet("border: 2px solid #2563EB;")
        super().focusInEvent(ev)

    def focusOutEvent(self, ev):
        self.setStyleSheet("border: none;")
        super().focusOutEvent(ev)

    def _on_scene_clicked(self, ev) -> None:
        if ev.isAccepted() or not self._duration_s: return
        t = self.plotItem.vb.mapSceneToView(ev.scenePos()).x()
        if not (0.0 <= t <= self._duration_s): return
        if ev.button() == Qt.MouseButton.LeftButton:
            self.seek_requested.emit(t)
        elif ev.button() == Qt.MouseButton.RightButton:
            self.waveform_right_clicked.emit(t)

    def set_waveform_data(self, peaks: np.ndarray, duration_s: float) -> None:
        self._duration_s = duration_s
        x = np.linspace(0, duration_s, len(peaks))
        self._upper.setData(x,  peaks)
        self._lower.setData(x, -peaks)
        mx = float(peaks.max()) if peaks.size else 1.0
        self.setXRange(0, duration_s, padding=0)
        self.setYRange(-mx * 1.15, mx * 1.15, padding=0)

    def clear_waveform(self) -> None:
        self._duration_s = 0.0
        self._upper.setData([], []); self._lower.setData([], [])
        self.setXRange(0, 1, padding=0); self.setYRange(-1, 1, padding=0)
        self._playhead.setPos(0)

    def set_playhead(self, pos_s: float) -> None:
        self._playhead.setPos(pos_s)

    def add_delete_region(self, rid: int, s: float, e: float) -> DeleteRegionItem:
        item = DeleteRegionItem(s, e)
        self.addItem(item); self._del_regions[rid] = item; return item

    def remove_delete_region(self, rid: int) -> None:
        if rid in self._del_regions:
            self.removeItem(self._del_regions.pop(rid))

    def add_vad_region(self, vid: int, s: float, e: float) -> VadRegionItem:
        item = VadRegionItem(s, e, self._duration_s)
        self.addItem(item); self._vad_regions[vid] = item; return item

    def remove_vad_region(self, vid: int) -> None:
        if vid in self._vad_regions:
            self.removeItem(self._vad_regions.pop(vid))

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if not self._duration_s: return
        sp = self.mapToScene(event.position().toPoint())
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_scene_x = sp.x()
        elif event.button() == Qt.MouseButton.LeftButton:
            t = self.plotItem.vb.mapSceneToView(sp).x()
            on_vad = any(lo <= t <= hi
                         for lo, hi in (v.getRegion() for v in self._vad_regions.values()))
            if not on_vad:
                self._left_press_x = sp.x(); self._left_dragging = False

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        sp = self.mapToScene(event.position().toPoint())
        vb = self.plotItem.vb
        if self._pan_scene_x is not None:
            lv = vb.mapSceneToView(QPointF(self._pan_scene_x, 0)).x()
            cv = vb.mapSceneToView(sp).x()
            self._apply_pan(lv - cv, vb); self._pan_scene_x = sp.x()
        elif self._left_press_x is not None:
            if not self._left_dragging and abs(sp.x() - self._left_press_x) > 3:
                self._left_dragging = True
            if self._left_dragging:
                lv = vb.mapSceneToView(QPointF(self._left_press_x, 0)).x()
                cv = vb.mapSceneToView(sp).x()
                self._apply_pan(lv - cv, vb); self._left_press_x = sp.x()

    def _apply_pan(self, d: float, vb) -> None:
        lo, hi = vb.viewRange()[0]
        span   = hi - lo
        new_lo = max(0.0, min(self._duration_s - span, lo + d))
        self.setXRange(new_lo, new_lo + span, padding=0)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_scene_x = None
        elif event.button() == Qt.MouseButton.LeftButton:
            self._left_press_x = None; self._left_dragging = False

    def wheelEvent(self, event):
        if not self._duration_s: return
        dy = event.angleDelta().y(); dx = event.angleDelta().x()
        pan = (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) or abs(dx) > abs(dy)
        if pan:
            delta = dx if abs(dx) > abs(dy) else dy
            if not delta: return
            lo, hi = self.plotItem.vb.viewRange()[0]
            span   = hi - lo
            shift  = span * 0.15 * (1 if delta > 0 else -1)
            new_lo = max(0.0, min(self._duration_s - span, lo + shift))
            self.setXRange(new_lo, new_lo + span, padding=0)
        else:
            if not dy: return
            sp     = self.mapToScene(event.position().toPoint())
            cx     = self.plotItem.vb.mapSceneToView(sp).x()
            lo, hi = self.plotItem.vb.viewRange()[0]
            span   = hi - lo
            factor = 0.7 if dy > 0 else 1.0 / 0.7
            ns     = max(1.0, min(self._duration_s, span * factor))
            ratio  = (cx - lo) / span if span else 0.5
            nl     = cx - ratio * ns
            nr     = nl + ns
            if nl < 0:  nl, nr = 0.0, ns
            if nr > self._duration_s: nr = self._duration_s; nl = max(0.0, nr - ns)
            self.setXRange(nl, nr, padding=0)


# ── Transcript view ────────────────────────────────────────────────────────────

class TranscriptView(QTextEdit):
    token_clicked       = Signal(int)
    range_selected      = Signal(int, int)
    right_clicked_token = Signal(int, QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.setStyleSheet("QTextEdit { border: none; padding: 8px; }")
        font = QFont(); font.setPixelSize(16); self.setFont(font)
        self._char_starts: list[int] = []
        self._char_ends:   list[int] = []

    def set_token_ranges(self, starts: list[int], ends: list[int]) -> None:
        self._char_starts = starts; self._char_ends = ends

    def token_at(self, char_pos: int) -> int:
        if not self._char_starts or char_pos < 0: return -1
        idx = bisect.bisect_right(self._char_starts, char_pos) - 1
        return idx if idx >= 0 and idx < len(self._char_ends) and char_pos < self._char_ends[idx] else -1

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton: super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if not self._char_starts:
            super().mouseReleaseEvent(event)
            return
        if event.button() == Qt.LeftButton:
            super().mouseReleaseEvent(event)
            cur = self.textCursor()
            if cur.hasSelection():
                s, e = cur.selectionStart(), cur.selectionEnd()
                if s >= 0 and e >= 0:
                    self.range_selected.emit(s, e)
            else:
                pos = self.cursorForPosition(event.position().toPoint()).position()
                if pos >= 0:
                    idx = self.token_at(pos)
                    if idx >= 0: self.token_clicked.emit(idx)
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # If Ctrl/Shift is held, ignore event to let it bubble up to MainWindow/Waveform
        if event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
            event.ignore()
        else:
            super().wheelEvent(event)

    def get_selection(self) -> tuple[int, int]:
        cur = self.textCursor()
        if not cur.hasSelection(): return -1, -1
        return cur.selectionStart(), cur.selectionEnd()

    def set_selection(self, start: int, end: int) -> None:
        cur = self.textCursor()
        if start < 0:
            cur.clearSelection()
        else:
            cur.setPosition(start)
            cur.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        self.setTextCursor(cur)

    def contextMenuEvent(self, event):
        if not self._char_starts: return
        idx = self.token_at(self.cursorForPosition(event.pos()).position())
        if idx < 0 and not self.textCursor().hasSelection():
            return
        self.right_clicked_token.emit(idx, event.globalPos())


# ── Segment bar ────────────────────────────────────────────────────────────────

class SegmentBar(QWidget):
    seek_clicked      = Signal(float)
    seg_right_clicked = Signal(int, QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self._segments: list[dict] = []
        self._x_min = 0.0
        self._x_max = 1.0
        self._hover_idx = -1
        self.setMouseTracking(True)
        self.setVisible(False)

    def set_segments(self, segments: list[dict]) -> None:
        self._segments = segments
        self.setVisible(bool(segments))
        self.update()

    def set_range(self, x_min: float, x_max: float) -> None:
        self._x_min = x_min
        self._x_max = x_max
        self.update()

    def _t_to_px(self, t: float) -> int:
        span = self._x_max - self._x_min
        if span <= 0: return 0
        return int((t - self._x_min) / span * self.width())

    def _px_to_t(self, px: int) -> float:
        span = self._x_max - self._x_min
        if self.width() <= 0: return self._x_min
        return self._x_min + px / self.width() * span

    def _seg_at(self, px: int) -> int:
        t = self._px_to_t(px)
        for i, seg in enumerate(self._segments):
            if float(seg["start"]) <= t < float(seg["end"]):
                return i
        if self._segments and t >= float(self._segments[-1]["end"]):
            return len(self._segments) - 1
        return -1

    def mousePressEvent(self, event):
        if not self._segments: return
        px = event.position().toPoint().x()
        if event.button() == Qt.MouseButton.LeftButton:
            self.seek_clicked.emit(self._px_to_t(px))
        elif event.button() == Qt.MouseButton.RightButton:
            idx = self._seg_at(px)
            if idx >= 0:
                self.seg_right_clicked.emit(idx, event.globalPosition().toPoint())

    def mouseMoveEvent(self, event):
        new_idx = self._seg_at(event.position().toPoint().x())
        if new_idx != self._hover_idx:
            self._hover_idx = new_idx
            self.update()

    def leaveEvent(self, event):
        self._hover_idx = -1
        self.update()

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPen, QFont as _QFont
        if not self._segments: return
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#111827"))
        w, h = self.width(), self.height()
        for i, seg in enumerate(self._segments):
            x1 = self._t_to_px(float(seg["start"]))
            x2 = self._t_to_px(float(seg["end"]))
            sw = x2 - x1
            if sw < 1: continue
            if i == self._hover_idx:
                p.fillRect(x1, 0, sw, h, QColor(99, 102, 241, 130))
            elif i % 2 == 0:
                p.fillRect(x1, 0, sw, h, QColor(99, 102, 241, 55))
            else:
                p.fillRect(x1, 0, sw, h, QColor(99, 102, 241, 20))
            if i > 0:
                p.setPen(QPen(QColor("#6366f1"), 1))
                p.drawLine(x1, 0, x1, h)
            if sw > 14:
                p.setPen(QPen(QColor("#e2e8f0")))
                font = _QFont(); font.setPixelSize(11)
                p.setFont(font)
                p.setClipRect(x1 + 3, 0, sw - 6, h)
                p.drawText(x1 + 3, 0, sw - 6, h,
                           Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                           seg["text"].strip())
                p.setClipping(False)
        p.end()


# ── Main window ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Podcast Editor")
        self.resize(800, 720)

        self._player    = QMediaPlayer()
        self._audio_out = QAudioOutput()
        self._player.setAudioOutput(self._audio_out)
        self._audio_out.setVolume(1.0)


        self._seeking          = False
        self._valid_tokens:    list[dict] = []
        self._token_starts_ms: list[int]  = []
        self._highlighted_idx  = -1

        self._delete_ranges:     list[dict] = []
        self._deleted_token_set: set[int]   = set()
        self._next_range_id = 0

        self._vad_regions:   list[dict] = []
        self._vad_token_set: set[int]   = set()
        self._next_vad_id = 0
        self._vad_visible = True

        self._audio_path = ""
        self._audio_name = ""
        self._duration_s = 0.0
        self._total_transcript_units = 0.0  # cjk chars + en_letters/5
        self._whisper_silence_s = 0.0       # Whisper token 間 gap 加總
        self._segments:    list[dict] = []
        self._token_edits: dict[int, str] = {}

        self._follow_mode = True

        self._search_results: list[int] = []
        self._search_idx = -1
        self._fillers_visible = False
        self._filler_words: list[str] = []
        self._filler_token_set: set[int] = set()
        self._vad_min_silence_ms = 300

        self._in_point_idx = -1
        self._out_point_idx = -1

        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []
        self._max_undo = 100

        self._export_time_secs = 0
        self._export_thread: QThread | None      = None
        self._export_worker: ExportWorker | None = None
        self._export_act:    QAction | None      = None
        self._exporting      = False

        self._playback_speed = 1.0

        self._session_restored = False

        self._wav_thread: QThread | None        = None
        self._wav_worker: WaveformWorker | None = None
        self._wav_job_id  = 0

        self._vad_thread: QThread | None    = None
        self._vad_worker: VadWorker | None  = None
        self._vad_job_id  = 0
        self._vad_btn:    QPushButton | None = None

        self._transcribe_thread: QThread | None          = None
        self._transcribe_worker: TranscribeWorker | None = None
        self._transcribe_job_id  = 0

        self._lufs_thread: QThread | None = None
        self._lufs_worker: LufsWorker | None = None

        cfg = FILE_MANAGER.load_config()
        self._model_repo: str = cfg.get("model_repo", MODELS[0]["repo"])

        self._build_ui()
        self._connect_signals()
        self._setup_shortcuts()
        QTimer.singleShot(0, self._try_restore_last_session)
        self._build_menu()

    def _build_ui(self):
        root   = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(6)
        self.setCentralWidget(root)

        self._scrubber = QSlider(Qt.Horizontal)
        self._scrubber.setRange(0, 0)
        layout.addWidget(self._scrubber)

        ctrl = QHBoxLayout()
        # 左側：播放控制
        self._play_btn = QPushButton("播放"); self._play_btn.setFixedWidth(55)
        self._play_btn.setToolTip("播放/暫停 (Space)")
        ctrl.addWidget(self._play_btn)

        self._speed_btn = QPushButton("1.0x"); self._speed_btn.setFixedWidth(50)
        self._speed_btn.setMenu(self._build_speed_menu())
        ctrl.addWidget(self._speed_btn)

        self._follow_btn = QPushButton("跟隨"); self._follow_btn.setFixedWidth(45)
        self._follow_btn.setCheckable(True)
        self._follow_btn.setChecked(True)
        self._follow_btn.setToolTip("播放時自動捲動波形圖")
        self._follow_btn.setStyleSheet(
            "QPushButton:checked { background-color: #2563EB; color: white; }")
        ctrl.addWidget(self._follow_btn)

        self._vol_slider = QSlider(Qt.Horizontal)
        self._vol_slider.setRange(0, 100); self._vol_slider.setValue(100)
        self._vol_slider.setFixedWidth(80); self._vol_slider.setToolTip("音量")
        ctrl.addWidget(self._vol_slider)

        self._vol_label = QLabel("100%")
        self._vol_label.setFixedWidth(40)
        ctrl.addWidget(self._vol_label)

        ctrl.addStretch()

        # 右側：分析功能切換
        self._vad_btn = QPushButton("偵測靜音"); self._vad_btn.setFixedWidth(80)
        self._vad_btn.setToolTip("偵測靜音區間；偵測後點擊可切換顯示/隱藏")
        self._vad_btn.setEnabled(False)
        ctrl.addWidget(self._vad_btn)

        self._filler_btn = QPushButton("標記贅字"); self._filler_btn.setFixedWidth(80)
        self._filler_btn.setToolTip("標記贅字；標記後點擊可切換顯示/隱藏")
        ctrl.addWidget(self._filler_btn)

        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        ctrl.addWidget(self._time_label)
        layout.addLayout(ctrl)

        wc = QWidget()
        wc_layout = QVBoxLayout(wc)
        wc_layout.setContentsMargins(0, 0, 0, 0)
        wc_layout.setSpacing(0)
        self._segment_bar = SegmentBar()
        wc_layout.addWidget(self._segment_bar)
        self._waveform = WaveformWidget()
        self._waveform.setFocusPolicy(Qt.StrongFocus)
        wc_layout.addWidget(self._waveform)
        self._wav_scrollbar = QScrollBar(Qt.Horizontal)
        self._wav_scrollbar.setRange(0, 0)
        self._wav_scrollbar.setEnabled(False)
        wc_layout.addWidget(self._wav_scrollbar)

        self._splitter = QSplitter(Qt.Vertical)
        self._splitter.addWidget(wc)
        self._transcript_view = TranscriptView()
        self._transcript_view.setFocusPolicy(Qt.StrongFocus)
        self._splitter.addWidget(self._transcript_view)
        self._splitter.setSizes([140, 520])
        self._splitter.setHandleWidth(6)
        layout.addWidget(self._splitter, stretch=1)

        self._search_container = QWidget()
        search_layout = QHBoxLayout(self._search_container)
        search_layout.setContentsMargins(0, 0, 0, 0)
        self._search_bar = QLineEdit()
        self._search_bar.setPlaceholderText("搜尋逐字稿… (Ctrl+F)")
        self._search_bar.setStyleSheet("padding: 4px 8px; font-size: 13px;")
        self._search_bar.returnPressed.connect(self._on_search)
        search_layout.addWidget(self._search_bar)
        
        self._search_prev_btn = QPushButton("上一筆"); self._search_prev_btn.clicked.connect(self._search_prev)
        self._search_next_btn = QPushButton("下一筆"); self._search_next_btn.clicked.connect(self._search_next)
        self._search_close_btn = QPushButton("關閉"); self._search_close_btn.clicked.connect(self._close_search)
        
        search_layout.addWidget(self._search_prev_btn)
        search_layout.addWidget(self._search_next_btn)
        search_layout.addWidget(self._search_close_btn)
        
        self._search_container.setVisible(False)
        layout.addWidget(self._search_container)

        info_row = QHBoxLayout()
        self._status_label = QLabel("請點擊 檔案 > 開啟音檔 來開始")
        self._status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._status_label.setStyleSheet("color: grey; font-size: 11px;")
        info_row.addWidget(self._status_label, stretch=1)

        self._lufs_label = QLabel("")
        self._lufs_label.setFixedWidth(70)
        self._lufs_label.setAlignment(Qt.AlignCenter)
        self._lufs_label.setStyleSheet("color: #9ca3af; font-size: 10px;")
        info_row.addWidget(self._lufs_label)

        self._wpm_label = QLabel("")
        self._wpm_label.setFixedWidth(80)
        self._wpm_label.setAlignment(Qt.AlignCenter)
        self._wpm_label.setStyleSheet("color: #9ca3af; font-size: 10px;")
        info_row.addWidget(self._wpm_label)

        layout.addLayout(info_row)

    def _connect_signals(self):
        self._play_btn.clicked.connect(self._toggle_play)
        self._vad_btn.clicked.connect(self._on_vad_btn_clicked)
        self._filler_btn.clicked.connect(self._on_filler_btn_clicked)
        self._follow_btn.toggled.connect(self._on_follow_toggled)
        self._vol_slider.valueChanged.connect(self._on_volume_changed)
        self._scrubber.sliderPressed.connect(self._on_scrubber_pressed)
        self._scrubber.sliderReleased.connect(self._on_scrubber_released)
        self._scrubber.valueChanged.connect(self._on_scrubber_value_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.playbackStateChanged.connect(self._on_state_changed)
        self._transcript_view.token_clicked.connect(self._on_token_clicked)
        self._transcript_view.range_selected.connect(self._on_range_selected)
        self._transcript_view.right_clicked_token.connect(self._on_right_click_token)
        self._waveform.seek_requested.connect(
            lambda t: self._player.setPosition(int(t * 1000)))
        self._waveform.range_changed.connect(self._on_waveform_range_changed)
        self._waveform.range_changed.connect(self._segment_bar.set_range)
        self._waveform.waveform_right_clicked.connect(self._on_waveform_right_click)
        self._segment_bar.seek_clicked.connect(
            lambda t: self._player.setPosition(int(t * 1000)))
        self._segment_bar.seg_right_clicked.connect(self._on_seg_bar_right_click)
        self._wav_scrollbar.valueChanged.connect(self._on_scrollbar_changed)
        self._transcript_view.installEventFilter(self)
        self.installEventFilter(self)

    def _build_menu(self):
        menu = self.menuBar()

        # ── 檔案 ──────────────────────────────────────────────────────────────
        file_menu = menu.addMenu("檔案")

        open_act = QAction("開啟音檔…", self); open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._open_file)
        file_menu.addAction(open_act)

        transcribe_act = QAction("AI 轉錄逐字稿…", self); transcribe_act.setShortcut("Ctrl+T")
        transcribe_act.triggered.connect(self._transcribe_audio)
        file_menu.addAction(transcribe_act)

        load_transcript_act = QAction("載入逐字稿…", self); load_transcript_act.setShortcut("Ctrl+L")
        load_transcript_act.triggered.connect(self._open_transcript_file)
        file_menu.addAction(load_transcript_act)

        file_menu.addSeparator()

        save_act = QAction("儲存進度…", self); save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self._save_episode_session)
        file_menu.addAction(save_act)

        restore_act = QAction("從手動存檔還原…", self)
        restore_act.triggered.connect(self._restore_from_manual_save)
        file_menu.addAction(restore_act)

        file_menu.addSeparator()

        self._export_act = QAction("匯出音檔…", self); self._export_act.setShortcut("Ctrl+E")
        self._export_act.triggered.connect(self._export)
        file_menu.addAction(self._export_act)

        txt_act = QAction("匯出逐字稿 (TXT)…", self)
        txt_act.triggered.connect(self._export_txt)
        file_menu.addAction(txt_act)

        srt_act = QAction("匯出字幕 (SRT)…", self)
        srt_act.triggered.connect(self._export_srt)
        file_menu.addAction(srt_act)

        file_menu.addSeparator()

        quit_act = QAction("離開", self); quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # ── 編輯 ──────────────────────────────────────────────────────────────
        edit_menu = menu.addMenu("編輯")

        undo_act = QAction("復原", self); undo_act.setShortcut("Ctrl+Z")
        undo_act.triggered.connect(self._undo)
        edit_menu.addAction(undo_act)

        redo_act = QAction("重做", self); redo_act.setShortcut("Ctrl+Shift+Z")
        redo_act.triggered.connect(self._redo)
        edit_menu.addAction(redo_act)

        edit_menu.addSeparator()

        filler_act = QAction("語速與贅字分析…", self)
        filler_act.triggered.connect(self._detect_filler_words)
        edit_menu.addAction(filler_act)

        # ── 設定 ──────────────────────────────────────────────────────────────
        pref_menu = menu.addMenu("設定")

        model_act = QAction("選擇模型…", self)
        model_act.triggered.connect(lambda: self._show_model_dialog())
        pref_menu.addAction(model_act)

        prompt_act = QAction("設定提示詞…", self)
        prompt_act.triggered.connect(self._show_prompt_settings)
        pref_menu.addAction(prompt_act)

        pref_menu.addSeparator()

        clear_cache_act = QAction("清除波形暫存檔…", self)
        clear_cache_act.triggered.connect(self._clear_waveform_cache)
        pref_menu.addAction(clear_cache_act)

        clear_dirs_act = QAction("清除其他音檔資料夾…", self)
        clear_dirs_act.triggered.connect(self._clear_other_project_dirs)
        pref_menu.addAction(clear_dirs_act)

    def _on_volume_changed(self, val: int):
        self._audio_out.setVolume(val / 100.0)
        self._vol_label.setText(f"{val}%")

    def _update_vad_btn_text(self) -> None:
        if not self._vad_regions:
            self._vad_btn.setText("偵測靜音")
        elif self._vad_visible:
            self._vad_btn.setText("隱藏靜音")
        else:
            self._vad_btn.setText("顯示靜音")

    def _update_filler_btn_text(self) -> None:
        if not self._filler_words:
            self._filler_btn.setText("標記贅字")
        elif self._fillers_visible:
            self._filler_btn.setText("隱藏贅字")
        else:
            self._filler_btn.setText("顯示贅字")

    def _on_vad_btn_clicked(self) -> None:
        if not self._vad_regions:
            self._run_vad()
        else:
            self._vad_visible = not self._vad_visible
            for item in self._waveform._vad_regions.values():
                item.setVisible(self._vad_visible)
            self._refresh_all_extra_selections()
            self._update_vad_btn_text()

    def _on_filler_btn_clicked(self) -> None:
        if not self._filler_words:
            self._show_filler_settings()
        else:
            self._fillers_visible = not self._fillers_visible
            self._update_filler_highlights()
            self._update_filler_btn_text()

    def _clear_waveform_cache(self) -> None:
        files = list(WAVEFORM_CACHE_DIR.glob("*.npz"))
        if not files:
            QMessageBox.information(self, "清除暫存檔", "暫存目錄沒有檔案。")
            return
        size_mb = sum(f.stat().st_size for f in files) / 1024 ** 2
        btn = QMessageBox.question(self, "清除暫存檔",
            f"確定刪除 {len(files)} 個波形暫存檔（共 {size_mb:.1f} MB）？")
        if btn == QMessageBox.Yes:
            for f in files:
                f.unlink(missing_ok=True)
            QMessageBox.information(self, "清除暫存檔", f"已刪除 {len(files)} 個暫存檔。")

    def _clear_other_project_dirs(self) -> None:
        current_name = Path(self._audio_path).stem if self._audio_path else None
        dirs = [d for d in FILE_MANAGER.base_dir.iterdir()
                if d.is_dir() and d.name != current_name and d.name != "waveform_cache"]
        if not dirs:
            QMessageBox.information(self, "清除資料夾", "沒有其他音檔資料夾。")
            return
        names = "\n".join(f"  {d.name}" for d in dirs)
        btn = QMessageBox.question(self, "清除其他音檔資料夾",
            f"確定刪除以下 {len(dirs)} 個資料夾？\n{names}")
        if btn == QMessageBox.Yes:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)
            QMessageBox.information(self, "清除資料夾", f"已刪除 {len(dirs)} 個資料夾。")

    def _show_filler_settings(self):
        cfg = FILE_MANAGER.load_config()
        current = cfg.get("filler_words", "嗯,呃,啊,那個,這個,然後,所以說")
        text, ok = QInputDialog.getMultiLineText(self, "設定贅字", "請輸入要標記的贅字（以逗號或換行分隔）：", current.replace(",", "\n"))
        if ok:
            words = [w.strip() for w in text.replace("\n", ",").split(",") if w.strip()]
            cfg["filler_words"] = ",".join(words)
            FILE_MANAGER.save_config(cfg)
            self._filler_words = words
            self._fillers_visible = True
            self._update_filler_highlights()
            self._update_filler_btn_text()

    def _update_filler_highlights(self):
        words = self._filler_words
        if not self._fillers_visible or not hasattr(self, "_valid_tokens") or not self._valid_tokens or not words:
            self._filler_token_set = set()
        else:
            self._filler_token_set = {
                i for i, t in enumerate(self._valid_tokens)
                if t["text"].strip().lower() in [w.lower() for w in words]
            }
        self._refresh_all_extra_selections()

    def _refresh_all_extra_selections(self):
        extra = []
        doc = self._transcript_view.document()
        if not doc: return
        
        vad_set = getattr(self, "_vad_token_set", set())
        filler_set = getattr(self, "_filler_token_set", set())
        
        # VAD Highlights (Yellow) - priority over filler
        for idx in (vad_set if self._vad_visible else set()):
            sel = QTextEdit.ExtraSelection()
            sel.format.setBackground(_VAD_BG)
            sel.format.setForeground(_VAD_FG)
            sel.cursor = QTextCursor(doc)
            sel.cursor.setPosition(self._transcript_view._char_starts[idx])
            sel.cursor.setPosition(self._transcript_view._char_ends[idx], QTextCursor.MoveMode.KeepAnchor)
            extra.append(sel)

        # Filler Highlights (Orange) - only if not VAD and visible
        for idx in (filler_set if self._fillers_visible else set()):
            if idx in vad_set: continue
            sel = QTextEdit.ExtraSelection()
            sel.format.setBackground(_FILLER_BG)
            sel.format.setForeground(_FILLER_FG)
            sel.cursor = QTextCursor(doc)
            sel.cursor.setPosition(self._transcript_view._char_starts[idx])
            sel.cursor.setPosition(self._transcript_view._char_ends[idx], QTextCursor.MoveMode.KeepAnchor)
            extra.append(sel)

        self._transcript_view.setExtraSelections(extra)

    def _clear_filler_highlights(self):
        self._filler_words = []
        self._fillers_visible = False
        self._update_filler_highlights()
        self._update_filler_btn_text()

    def _detect_filler_words(self):
        if not self._valid_tokens:
            QMessageBox.warning(self, "分析", "請先載入逐字稿")
            return
        cfg = FILE_MANAGER.load_config()
        all_fillers = [w.strip() for w in cfg.get("filler_words", "嗯,呃,啊,那個,這個,然後,所以說").split(",") if w.strip()]
        counts: dict[str, int] = {}
        cjk = 0; en = 0
        for tok in self._valid_tokens:
            t_str = tok["text"].strip()
            for c in t_str:
                if '一' <= c <= '鿿' or '㐀' <= c <= '䶿':
                    cjk += 1
                elif c.isalpha() and ord(c) < 128:
                    en += 1
            if t_str.lower() in all_fillers:
                counts[t_str.lower()] = counts.get(t_str.lower(), 0) + 1

        units = cjk + en / 5.0
        dur = self._duration_s if self._duration_s > 0 else self._waveform._duration_s
        if dur <= 0: dur = self._valid_tokens[-1]["end"] if self._valid_tokens else 0
        silence = (sum(r["end"] - r["start"] for r in self._vad_regions)
                   if self._vad_regions else self._whisper_silence_s)
        speaking_s = max(dur - silence, 1.0)
        wpm = (units / speaking_s * 60) if dur > 0 else 0

        report = []
        report.append(f"中文字數：{cjk}　英文字數：{en}（≈ {en/5:.0f} 字換算）")
        report.append(f"音檔時長：{int(dur//60)}:{int(dur%60):02d}　停頓：{int(silence)}s")
        report.append(f"整體語速：{wpm:.0f} 字/分鐘")
        report.append("")
        
        if counts:
            report.append("偵測到以下常見贅字：")
            for w, c in sorted(counts.items(), key=lambda x: -x[1]):
                report.append(f"  「{w}」: {c} 次")
        else:
            report.append("未偵測到常見的口頭禪/贅字。")
            
        QMessageBox.information(self, "語速與贅字分析", "\n".join(report))

    def _on_search(self):
        text = self._search_bar.text().strip()
        old_results = self._search_results if hasattr(self, "_search_results") else []
        self._search_results = []
        self._search_idx = -1
        
        if not text or not self._valid_tokens:
            self._transcript_view.setUpdatesEnabled(False)
            for idx in old_results:
                self._apply_single_format(idx)
            self._transcript_view.setUpdatesEnabled(True)
            self._status_label.setText("搜尋已清除")
            return
            
        self._search_results = [
            i for i, tok in enumerate(self._valid_tokens)
            if text.lower() in tok["text"].lower()
        ]
        
        self._transcript_view.setUpdatesEnabled(False)
        for idx in set(old_results + self._search_results):
            self._apply_single_format(idx)
        self._transcript_view.setUpdatesEnabled(True)
        
        if self._search_results:
            self._search_idx = 0
            self._jump_to_search_result(0)
            self._update_search_status()
        else:
            self._status_label.setText(f"搜尋：未找到「{text}」")

    def _update_search_status(self):
        if self._search_results:
            self._status_label.setText(f"搜尋：第 {self._search_idx + 1} 筆，共 {len(self._search_results)} 筆")

    def _close_search(self):
        self._search_container.setVisible(False)
        self._search_bar.clear()
        self._on_search()

    def _jump_to_search_result(self, n: int):
        if not self._search_results or n >= len(self._search_results):
            return
        idx = self._search_results[n]
        self._player.setPosition(int(self._valid_tokens[idx]["start"] * 1000))
        self._apply_single_format(idx)
        self._scroll_to_token(idx)

    def _search_next(self):
        if not self._search_results: return
        old_idx = self._search_idx
        self._search_idx = (self._search_idx + 1) % len(self._search_results)
        self._apply_single_format(self._search_results[old_idx])
        self._jump_to_search_result(self._search_idx)
        self._update_search_status()

    def _search_prev(self):
        if not self._search_results: return
        old_idx = self._search_idx
        self._search_idx = (self._search_idx - 1) % len(self._search_results)
        self._apply_single_format(self._search_results[old_idx])
        self._jump_to_search_result(self._search_idx)
        self._update_search_status()

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+F"), self).activated.connect(self._toggle_search)
        QShortcut(QKeySequence("F3"), self).activated.connect(self._search_next)
        QShortcut(QKeySequence("Shift+F3"), self).activated.connect(self._search_prev)
        QShortcut(QKeySequence("Escape"), self).activated.connect(self._cancel_selection_or_search)
        
        s_del = QShortcut(QKeySequence("Ctrl+D"), self)
        s_del.setContext(Qt.ApplicationShortcut)
        s_del.activated.connect(self._delete_current_selection)

        s_left = QShortcut(QKeySequence("["), self._waveform)
        s_left.setContext(Qt.WidgetWithChildrenShortcut)
        s_left.activated.connect(lambda: self._nudge_selection(-1))
        
        s_right = QShortcut(QKeySequence("]"), self._waveform)
        s_right.setContext(Qt.WidgetWithChildrenShortcut)
        s_right.activated.connect(lambda: self._nudge_selection(1))

    def _toggle_search(self):
        self._search_container.setVisible(not self._search_container.isVisible())
        if self._search_container.isVisible():
            self._search_bar.setFocus()
        else:
            self._close_search()

    def _cancel_selection_or_search(self):
        self._close_search()
        self._highlighted_idx = -1

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Space:
                if not isinstance(QApplication.focusWidget(), QLineEdit):
                    self._toggle_play()
                    return True
        return super().eventFilter(obj, event)

    def _set_in_point(self):
        if not self._valid_tokens: return
        pos_ms = self._player.position()
        idx = bisect.bisect_right(self._token_starts_ms, pos_ms) - 1
        if idx < 0: return
        self._in_point_idx = idx
        self._status_label.setText(f"In 點設於: {self._valid_tokens[idx]['text']}")

    def _set_out_point(self):
        if not self._valid_tokens: return
        pos_ms = self._player.position()
        idx = bisect.bisect_left(self._token_starts_ms, pos_ms)
        if idx >= len(self._valid_tokens): idx = len(self._valid_tokens) - 1
        self._out_point_idx = idx
        self._status_label.setText(f"Out 點設於: {self._valid_tokens[idx]['text']}")
        if hasattr(self, "_in_point_idx") and self._in_point_idx >= 0 and self._in_point_idx < idx:
            st, et = self._in_point_idx, idx
            ss = self._valid_tokens[st]["start"]
            es = self._valid_tokens[et]["end"]
            self._merge_into_delete(ss, es)
            self._in_point_idx = self._out_point_idx = -1

    def _nudge_selection(self, direction: int):
        if not self._delete_ranges:
            return
        dr = self._delete_ranges[-1]
        st, et = dr["start_idx"], dr["end_idx"]
        if direction < 0 and st > 0:
            st -= 1
        elif direction > 0 and et < len(self._valid_tokens) - 1:
            et += 1
        else:
            return
        self._push_undo("nudge", self._make_snapshot())
        for i in range(dr["start_idx"], dr["end_idx"] + 1):
            if not any(o["start_idx"] <= i <= o["end_idx"] for o in self._delete_ranges if o["id"] != dr["id"]):
                self._deleted_token_set.discard(i)
        self._batch_apply_format(dr["start_idx"], dr["end_idx"], False)
        dr["start_idx"] = st; dr["end_idx"] = et
        dr["start"] = self._valid_tokens[st]["start"]
        dr["end"] = self._valid_tokens[et]["end"]
        for i in range(st, et + 1): self._deleted_token_set.add(i)
        self._batch_apply_format(st, et, True)
        self._waveform.remove_delete_region(dr["id"])
        item = self._waveform.add_delete_region(dr["id"], dr["start"], dr["end"])
        item.sig_right_click.connect(lambda pos, r=dr["id"]: self._on_delete_region_right_click(r, pos))
        self._update_status()
        self._save_session()

    def _build_speed_menu(self) -> QMenu:
        menu = QMenu(self)
        for rate in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            act = QAction(f"{rate:.2g}x", self)
            act.setCheckable(True)
            act.setChecked(abs(rate - self._playback_speed) < 0.01)
            act.triggered.connect(lambda checked, r=rate: self._set_speed(r))
            menu.addAction(act)
        return menu

    def _set_speed(self, rate: float):
        self._playback_speed = rate
        self._speed_btn.setText(f"{rate:.2g}x")
        self._player.setPlaybackRate(rate)
        self._speed_btn.setMenu(self._build_speed_menu())

    def _start_transcribe_with_dialog(self, audio_path: str, output_path: str):
        result = self._show_model_dialog(audio_duration_s(audio_path))
        if result is None:
            return
        precision, prompt = result
        word_ts = precision == "word"
        self._start_transcribe_worker(audio_path, output_path,
                                      word_timestamps=word_ts, initial_prompt=prompt)

    def _show_model_dialog(self, dur_s: float = 0.0):
        dlg = ModelSelectDialog(self._model_repo, dur_s, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        self._model_repo = dlg.selected_repo()
        prompt = dlg.selected_prompt()
        cfg = FILE_MANAGER.load_config()
        cfg["model_repo"] = self._model_repo
        cfg["initial_prompt"] = prompt
        FILE_MANAGER.save_config(cfg)
        return dlg.selected_precision(), prompt

    def _clear_regions(self) -> None:
        for rid in list(self._waveform._del_regions): self._waveform.remove_delete_region(rid)
        for vid in list(self._waveform._vad_regions): self._waveform.remove_vad_region(vid)
        self._delete_ranges.clear();    self._deleted_token_set.clear(); self._next_range_id = 0
        self._vad_regions.clear();      self._vad_token_set.clear();     self._next_vad_id   = 0
        self._vad_visible = True
        if self._vad_btn is not None:
            self._update_vad_btn_text()

    def _clear_transcript(self) -> None:
        self._transcript_view.clear()
        self._transcript_view.set_token_ranges([], [])
        self._transcript_view.setPlaceholderText("")
        self._valid_tokens.clear(); self._token_starts_ms.clear()
        self._highlighted_idx = -1
        self._total_transcript_units = 0.0
        self._whisper_silence_s = 0.0
        self._wpm_label.setText("")
        self._segments.clear()
        self._token_edits.clear()
        self._segment_bar.set_segments([])

    def _populate_transcript(self, data: dict) -> None:
        self._clear_regions(); self._clear_transcript()
        view   = self._transcript_view
        full_text = ""
        token_info = []
        for tok in data.get("tokens", []):
            if tok.get("start") is None or tok.get("end") is None: continue
            text = tok["text"]
            start_pos = len(full_text)
            full_text += text
            end_pos = len(full_text)
            token_info.append((start_pos, end_pos, tok))
        
        cursor = QTextCursor(view.document())
        blk = QTextBlockFormat(); blk.setLineHeight(150, 1)
        cursor.setBlockFormat(blk)
        cursor.insertText(full_text)
        
        cs = [ti[0] for ti in token_info]
        ce = [ti[1] for ti in token_info]
        self._valid_tokens = [ti[2] for ti in token_info]
        self._token_starts_ms = [int(ti[2]["start"] * 1000) for ti in token_info]
        
        view.set_token_ranges(cs, ce)
        
        cjk = 0; en = 0
        for t in self._valid_tokens:
            for c in t["text"]:
                if '一' <= c <= '鿿' or '㐀' <= c <= '䶿':
                    cjk += 1
                elif c.isalpha() and ord(c) < 128:
                    en += 1
        self._total_transcript_units = cjk + en / 5.0
        self._whisper_silence_s = sum(
            max(0.0, self._valid_tokens[i + 1]["start"] - self._valid_tokens[i]["end"])
            for i in range(len(self._valid_tokens) - 1)
        )
        self._update_wpm_label()

        cfg = FILE_MANAGER.load_config()
        self._filler_words = [w.strip() for w in cfg.get("filler_words", "嗯,呃,啊,那個,這個,然後,所以說").split(",") if w.strip()]
        self._filler_token_set = {
            i for i, t in enumerate(self._valid_tokens)
            if t["text"].strip() in self._filler_words
        }

        self._segments = data.get("segments", [])
        self._segment_bar.set_segments(self._segments)
        lo, hi = self._waveform.plotItem.vb.viewRange()[0]
        self._segment_bar.set_range(lo, hi)

        if not self._audio_name:
            ap = data.get("audio_path", "")
            self._audio_name = Path(ap).name if ap else ""
        self._update_status()
        self._restore_session_if_ready()

    def _start_waveform_worker(self, path: str) -> None:
        self._wav_job_id += 1; job_id = self._wav_job_id
        self._status_label.setText("正在分析波形…")
        self._waveform.clear_waveform()
        self._wav_scrollbar.setRange(0, 0); self._wav_scrollbar.setEnabled(False)
        if qt_obj_alive(self._wav_thread) and self._wav_thread.isRunning():
            self._wav_thread.requestInterruption(); self._wav_thread.quit()
        else:
            self._wav_thread = self._wav_worker = None
        t = QThread(); w = WaveformWorker(job_id, path, WAVEFORM_CACHE_DIR)
        w.moveToThread(t); t.started.connect(w.run)
        w.finished.connect(self._on_waveform_ready); w.finished.connect(t.quit)
        t.finished.connect(self._on_waveform_thread_done)
        t.finished.connect(w.deleteLater); t.finished.connect(t.deleteLater)
        self._wav_thread = t; self._wav_worker = w; t.start()

    def _on_waveform_ready(self, job_id: int, peaks: np.ndarray, dur: float) -> None:
        if job_id != self._wav_job_id: return
        self._waveform.set_waveform_data(peaks, dur)
        self._reset_scrollbar(dur); self._update_status()
        self._update_wpm_label()

    def _on_waveform_thread_done(self) -> None:
        self._wav_thread = self._wav_worker = None

    def _cancel_vad_if_running(self) -> None:
        if qt_obj_alive(self._vad_thread) and self._vad_thread.isRunning():
            self._vad_job_id += 1
            self._vad_thread.requestInterruption(); self._vad_thread.quit()
        else:
            self._vad_thread = self._vad_worker = None

    def _start_transcribe_worker(self, path: str, output_path: str,
                                  word_timestamps: bool = False, initial_prompt: str = "") -> None:
        self._transcribe_job_id += 1; job_id = self._transcribe_job_id
        self._transcript_view.setPlaceholderText("正在轉錄…")
        self._status_label.setText("正在啟動轉錄…")
        if qt_obj_alive(self._transcribe_thread) and self._transcribe_thread.isRunning():
            self._transcribe_thread.requestInterruption(); self._transcribe_thread.quit()
        else:
            self._transcribe_thread = self._transcribe_worker = None
        t = QThread()
        w = TranscribeWorker(job_id, path, self._model_repo, output_path, word_timestamps, initial_prompt)
        w.moveToThread(t); t.started.connect(w.run)
        w.progress.connect(self._on_transcribe_progress)
        w.finished.connect(self._on_transcribe_finished)
        w.error.connect(self._on_transcribe_error)
        w.finished.connect(t.quit); w.error.connect(t.quit)
        t.finished.connect(self._on_transcribe_thread_done)
        t.finished.connect(w.deleteLater); t.finished.connect(t.deleteLater)
        self._transcribe_thread = t; self._transcribe_worker = w
        self._status_label.setText("正在啟動轉錄…（可能需要幾秒載入模型）")
        QApplication.processEvents()
        t.start()

    def _on_transcribe_progress(self, job_id: int, msg: str) -> None:
        if job_id == self._transcribe_job_id:
            self._status_label.setText(msg)
            QApplication.processEvents()

    def _on_transcribe_finished(self, job_id: int, data: dict) -> None:
        if job_id != self._transcribe_job_id: return
        self._status_label.setText("正在載入逐字稿…")
        QApplication.processEvents()
        self._populate_transcript(data)

    def _on_transcribe_error(self, job_id: int, msg: str) -> None:
        if job_id != self._transcribe_job_id: return
        QMessageBox.critical(self, "轉錄失敗", msg)
        self._status_label.setText("轉錄失敗")

    def _on_transcribe_thread_done(self) -> None:
        self._transcribe_thread = self._transcribe_worker = None

    def _start_lufs_worker(self, path: str):
        self._lufs_label.setText("計算 LUFS…")
        if qt_obj_alive(self._lufs_thread) and self._lufs_thread.isRunning():
            self._lufs_thread.requestInterruption(); self._lufs_thread.quit()
        t = QThread(); w = LufsWorker(path)
        w.moveToThread(t); t.started.connect(w.run)
        w.finished.connect(self._lufs_label.setText)
        w.finished.connect(t.quit)
        t.finished.connect(w.deleteLater); t.finished.connect(t.deleteLater)
        self._lufs_thread = t; self._lufs_worker = w; t.start()

    def _reset_scrollbar(self, dur: float) -> None:
        total = int(dur * 1000)
        self._wav_scrollbar.setRange(0, total)
        self._wav_scrollbar.setPageStep(total)
        self._wav_scrollbar.setEnabled(False)

    def _on_waveform_range_changed(self, x_min: float, x_max: float) -> None:
        dur = self._waveform._duration_s
        if not dur: return
        total = int(dur * 1000)
        span  = int((x_max - x_min) * 1000)
        self._wav_scrollbar.blockSignals(True)
        self._wav_scrollbar.setRange(0, max(0, total - span))
        self._wav_scrollbar.setPageStep(span)
        self._wav_scrollbar.setValue(int(x_min * 1000))
        self._wav_scrollbar.setEnabled(span < total)
        self._wav_scrollbar.blockSignals(False)

    def _on_scrollbar_changed(self, value: int) -> None:
        if not self._waveform._duration_s: return
        lo, hi = self._waveform.plotItem.vb.viewRange()[0]
        span   = hi - lo
        new_lo = value / 1000.0
        self._waveform.setXRange(new_lo, new_lo + span, padding=0)

    def _apply_single_format(self, idx: int) -> None:
        fmt = QTextCharFormat()
        is_search = False
        is_search_active = False
        if hasattr(self, "_search_results") and self._search_results:
            is_search = idx in self._search_results
            if is_search and self._search_idx >= 0 and self._search_idx < len(self._search_results):
                is_search_active = (idx == self._search_results[self._search_idx])
        
        if idx == self._highlighted_idx:
            fmt.setBackground(_HL_BG);  fmt.setForeground(_HL_FG)
        elif is_search_active:
            fmt.setBackground(_SEARCH_ACTIVE_BG); fmt.setForeground(_SEARCH_ACTIVE_FG)
        elif is_search:
            fmt.setBackground(_SEARCH_BG); fmt.setForeground(_SEARCH_FG)
        elif idx in self._deleted_token_set:
            fmt.setBackground(_DEL_BG); fmt.setForeground(_DEL_FG)
        elif idx in self._vad_token_set and self._vad_visible:
            fmt.setBackground(_VAD_BG); fmt.setForeground(_VAD_FG)
        elif idx in self._filler_token_set and self._fillers_visible:
            fmt.setBackground(_FILLER_BG); fmt.setForeground(_FILLER_FG)
        else:
            fmt.clearBackground(); fmt.clearForeground()
            
        view = self._transcript_view
        cur  = QTextCursor(view.document())
        cur.setPosition(view._char_starts[idx])
        cur.setPosition(view._char_ends[idx], QTextCursor.MoveMode.KeepAnchor)
        cur.setCharFormat(fmt)

    def _batch_apply_format(self, s: int, e: int, deleted: bool) -> None:
        view = self._transcript_view
        fmt  = QTextCharFormat()
        if deleted:
            fmt.setBackground(_DEL_BG); fmt.setForeground(_DEL_FG)
        else:
            fmt.clearBackground(); fmt.clearForeground()
        cur = QTextCursor(view.document())
        cur.setPosition(view._char_starts[s])
        cur.setPosition(view._char_ends[e], QTextCursor.MoveMode.KeepAnchor)
        cur.setCharFormat(fmt)
        if s <= self._highlighted_idx <= e:
            self._apply_single_format(self._highlighted_idx)

    def _on_range_selected(self, sc: int, ec: int) -> None:
        # Just store the selection state, don't auto-delete
        # The Cmd+D shortcut will call _delete_current_selection
        pass

    def _delete_current_selection(self):
        view = self._transcript_view
        sc, ec = view.get_selection()
        if sc < 0 or ec < 0:
            if self._highlighted_idx >= 0:
                # Handle highlighted token (single click)
                idx = self._highlighted_idx
                self._add_delete_range_idx(idx, idx)
                self._highlighted_idx = -1
            return
            
        st = view.token_at(sc)
        et = view.token_at(ec - 1)
        if et < 0: et = view.token_at(ec)
        if st < 0 or et < 0 or st > et: return
        
        self._add_delete_range_idx(st, et)
        view.set_selection(-1, -1)

    def _add_delete_range_idx(self, st: int, et: int):
        ss = self._valid_tokens[st]["start"]
        es = self._valid_tokens[et]["end"]
        self._merge_into_delete(ss, es)

    def _remove_delete_range(self, idx: int) -> None:
        dr = self._delete_ranges.pop(idx)
        st, et = dr["start_idx"], dr["end_idx"]
        for i in range(st, et + 1):
            if not any(o["start_idx"] <= i <= o["end_idx"] for o in self._delete_ranges):
                self._deleted_token_set.discard(i)
        self._batch_apply_format(st, et, False)
        for o in self._delete_ranges:
            os, oe = max(o["start_idx"], st), min(o["end_idx"], et)
            if os <= oe: self._batch_apply_format(os, oe, True)
        self._waveform.remove_delete_region(dr["id"]); self._update_status()

    def _delete_range_by_idx(self, idx: int) -> None:
        self._push_undo("remove_delete", self._make_snapshot())
        self._remove_delete_range(idx)
        self._save_session()

    def _on_right_click_token(self, idx: int, gpos: QPoint) -> None:
        if not self._valid_tokens: return
        menu = QMenu(self)

        # 0. Token text edit
        if idx >= 0:
            menu.addAction("編輯此詞語…").triggered.connect(
                lambda: self._edit_token_text(idx))
            menu.addSeparator()

        # 1. Check for text selection
        sc, ec = self._transcript_view.get_selection()
        if sc >= 0 and ec >= 0:
            act_del = menu.addAction("確認刪除選取範圍 (Confirm Delete)")
            act_del.triggered.connect(self._delete_current_selection)
            menu.addSeparator()

        # 2. Case: Deleted token
        if idx >= 0 and idx < len(self._valid_tokens) and idx in self._deleted_token_set:
            dr = next((d for d in self._delete_ranges if d["start_idx"] <= idx <= d["end_idx"]), None)
            if dr:
                menu.addAction("恢復此音訊 (Restore Audio)").triggered.connect(
                    lambda: self._restore_delete_range_by_id(dr["id"]))
                menu.addAction("恢復為靜音區間 (Restore as Mute)").triggered.connect(
                    lambda: self._restore_as_vad_by_id(dr["id"]))
                menu.addSeparator()

        # 3. Case: VAD token (yellow highlight)
        if idx >= 0 and idx < len(self._valid_tokens) and idx in self._vad_token_set:
            vr = next((r for r in self._vad_regions if r["start_tok"] <= idx <= r["end_tok"]), None)
            if vr:
                menu.addAction("確認刪除此靜音區間").triggered.connect(
                    lambda: self._confirm_vad(vr["id"]))
                menu.addAction("移除此靜音標記 (Ignore)").triggered.connect(
                    lambda: self._ignore_vad(vr["id"]))
                menu.addSeparator()
        
        if not menu.isEmpty():
            menu.exec(gpos)


    def _restore_delete_range_by_id(self, rid: int):
        idx = next((i for i, d in enumerate(self._delete_ranges) if d["id"] == rid), None)
        if idx is not None:
            self._remove_delete_range(idx)
            self._save_session()

    def _restore_as_vad_by_id(self, rid: int) -> None:
        idx = next((i for i, d in enumerate(self._delete_ranges) if d["id"] == rid), None)
        if idx is not None:
            self._push_undo("restore_as_vad", self._make_snapshot())
            self._restore_as_vad(idx)

    def _restore_as_vad(self, idx: int) -> None:
        dr = self._delete_ranges[idx]
        ss, es = dr["start"], dr["end"]
        self._remove_delete_range(idx)
        vid = self._next_vad_id; self._next_vad_id += 1
        item = self._waveform.add_vad_region(vid, ss, es)
        item.sig_right_click.connect(lambda pos, v=vid: self._on_vad_right_click(v, pos))
        item.sigRegionChangeFinished.connect(lambda _, v=vid: self._on_vad_boundary_changed(v))
        st, et = self._find_tokens_in_range(ss, es)
        self._vad_regions.append({"id": vid, "start": ss, "end": es,
                                  "start_tok": st, "end_tok": et})
        self._set_vad_tokens(st, et, True)
        self._update_status()
        self._save_session()

    # ── VAD ─────────────────────────────────────────────────────────────────────

    def _run_vad(self) -> None:
        if not self._audio_path: return
        if qt_obj_alive(self._vad_thread) and self._vad_thread.isRunning(): return
        if not qt_obj_alive(self._vad_thread): self._vad_thread = self._vad_worker = None
        
        # Show VAD duration dialog
        durations = [0.3, 0.5, 1.0, 1.5, 2.0]
        dur_labels = ["0.3 秒", "0.5 秒", "1.0 秒", "1.5 秒", "2.0 秒"]
        selected, ok = QInputDialog.getItem(self, "靜音偵測參數", "最小靜音時長：", dur_labels, 1, False)
        if not ok: return
        min_silence = durations[dur_labels.index(selected)]
        self._vad_min_silence_ms = int(min_silence * 1000)
        
        # 清除贅字標記，避免與 VAD 混合
        self._filler_token_set = set()
        self._fillers_visible = False
        
        self._vad_job_id += 1; job_id = self._vad_job_id
        self._vad_btn.setEnabled(False)
        self._status_label.setText("正在啟動 VAD…")
        dur = self._duration_s or self._player.duration() / 1000.0
        t = QThread()
        w = VadWorker(job_id, self._audio_path, dur,
                      threshold=0.5, min_silence_ms=self._vad_min_silence_ms, min_speech_ms=100)
        w.moveToThread(t); t.started.connect(w.run)
        w.status.connect(self._on_vad_status); w.finished.connect(self._on_vad_finished)
        w.error.connect(self._on_vad_error)
        w.finished.connect(t.quit); w.error.connect(t.quit)
        t.finished.connect(self._on_vad_thread_done)
        t.finished.connect(w.deleteLater); t.finished.connect(t.deleteLater)
        self._vad_thread = t; self._vad_worker = w; t.start()

    def _on_vad_status(self, job_id: int, msg: str) -> None:
        if job_id == self._vad_job_id: self._status_label.setText(msg)

    def _on_vad_finished(self, job_id: int, silence: list) -> None:
        if job_id != self._vad_job_id: return
        self._status_label.setText(f"VAD 完成，找到 {len(silence)} 個靜音區間")
        QApplication.processEvents()
        min_silence_s = self._vad_min_silence_ms / 1000.0
        for seg in silence:
            ss, es = seg["start"], seg["end"]
            if es - ss < min_silence_s: continue
            if any(not (es <= d["start"] or ss >= d["end"]) for d in self._delete_ranges): continue
            if any(not (es <= r["start"] or ss >= r["end"])   for r in self._vad_regions): continue
            vid = self._next_vad_id; self._next_vad_id += 1
            item = self._waveform.add_vad_region(vid, ss, es)
            item.sig_right_click.connect(lambda pos, v=vid: self._on_vad_right_click(v, pos))
            item.sigRegionChangeFinished.connect(lambda _, v=vid: self._on_vad_boundary_changed(v))
            st, et = self._find_tokens_in_range(ss, es)
            self._vad_regions.append({"id": vid, "start": ss, "end": es,
                                      "start_tok": st, "end_tok": et})
            if st >= 0 and et >= 0:
                for i in range(st, et + 1): self._vad_token_set.add(i)
        
        self._refresh_all_extra_selections()
        self._update_status()
        self._save_session()
        self._vad_visible = True
        self._update_vad_btn_text()
        QApplication.processEvents()

    def _on_vad_error(self, job_id: int, msg: str) -> None:
        if job_id == self._vad_job_id: QMessageBox.critical(self, "VAD 失敗", msg)

    def _on_vad_thread_done(self) -> None:
        self._vad_thread = self._vad_worker = None
        self._vad_btn.setEnabled(True)

    def _confirm_vad(self, vid: int) -> None:
        region = next((r for r in self._vad_regions if r["id"] == vid), None)
        if region is None: return
        item = self._waveform._vad_regions.get(vid)
        ss, es = item.getRegion() if item else (region["start"], region["end"])
        dur = self._duration_s or self._player.duration() / 1000.0
        if ss < 0.1: ss = 0.0
        if es > dur - 0.1: es = dur
        self._merge_into_delete(ss, es)

    def _on_delete_region_right_click(self, rid: int, gpos: QPoint) -> None:
        idx = next((i for i, d in enumerate(self._delete_ranges) if d["id"] == rid), None)
        if idx is None: return
        menu = QMenu(self)
        menu.addAction("恢復此音訊 (Restore Audio)").triggered.connect(
            lambda: self._delete_range_by_idx(idx))
        menu.addAction("恢復為靜音區間 (Restore as Mute)").triggered.connect(
            lambda: self._restore_as_vad_by_id(rid))
        menu.exec(gpos)

    def _on_vad_right_click(self, vid: int, gpos: QPoint) -> None:
        menu = QMenu(self)
        menu.addAction("確認刪除").triggered.connect(lambda: self._confirm_vad(vid))
        menu.addAction("忽略此區間").triggered.connect(lambda: self._ignore_vad(vid))
        menu.exec(gpos)

    def _ignore_vad(self, vid: int) -> None:
        self._remove_vad_region_by_id(vid)
        self._update_status()
        self._save_session()

    def _remove_vad_region_by_id(self, vid: int) -> None:
        idx = next((i for i, r in enumerate(self._vad_regions) if r["id"] == vid), None)
        if idx is None: return
        r = self._vad_regions.pop(idx)
        self._waveform.remove_vad_region(vid)
        self._set_vad_tokens(r["start_tok"], r["end_tok"], False)

    def _on_vad_boundary_changed(self, vid: int) -> None:
        region = next((r for r in self._vad_regions if r["id"] == vid), None)
        item   = self._waveform._vad_regions.get(vid)
        if region is None or item is None: return
        ss, es = item.getRegion()
        self._set_vad_tokens(region["start_tok"], region["end_tok"], False)
        region["start"] = ss; region["end"] = es
        ns, ne = self._find_tokens_in_range(ss, es)
        region["start_tok"] = ns; region["end_tok"] = ne
        self._set_vad_tokens(ns, ne, True)

    def _set_vad_tokens(self, st: int, et: int, add: bool) -> None:
        if st < 0 or et < 0: return
        for i in range(st, et + 1):
            if add:
                self._vad_token_set.add(i)
            else:
                if not any(r["start_tok"] <= i <= r["end_tok"] for r in self._vad_regions):
                    self._vad_token_set.discard(i)
        self._refresh_all_extra_selections()

    def _find_tokens_in_range(self, ss: float, es: float) -> tuple[int, int]:
        sms, ems = int(ss * 1000), int(es * 1000)
        idx = max(0, bisect.bisect_left(self._token_starts_ms, sms) - 1)
        first = last = -1
        for i in range(idx, len(self._valid_tokens)):
            t = self._valid_tokens[i]
            ts, te = int(t["start"] * 1000), int(t["end"] * 1000)
            if ts >= ems: break
            if te > sms:
                if first < 0: first = i
                last = i
        return first, last

    def _merge_into_delete(self, ss: float, es: float) -> None:
        # Expand range to absorb all overlapping regions (iterate until stable)
        changed = True
        while changed:
            changed = False
            for d in self._delete_ranges:
                if not (es <= d["start"] or ss >= d["end"]):
                    nss, nes = min(ss, d["start"]), max(es, d["end"])
                    if nss < ss or nes > es:
                        ss, es, changed = nss, nes, True
            for r in self._vad_regions:
                if not (es <= r["start"] or ss >= r["end"]):
                    nss, nes = min(ss, r["start"]), max(es, r["end"])
                    if nss < ss or nes > es:
                        ss, es, changed = nss, nes, True

        self._push_undo("add_delete", self._make_snapshot())

        del_ids = [d["id"] for d in self._delete_ranges
                   if not (es <= d["start"] or ss >= d["end"])]
        for did in del_ids:
            idx = next((i for i, d in enumerate(self._delete_ranges) if d["id"] == did), None)
            if idx is not None:
                self._remove_delete_range(idx)

        vad_ids = [r["id"] for r in self._vad_regions
                   if not (es <= r["start"] or ss >= r["end"])]
        for vid in vad_ids:
            self._remove_vad_region_by_id(vid)

        rid = self._next_range_id; self._next_range_id += 1
        st, et = self._find_tokens_in_range(ss, es)
        self._delete_ranges.append({"id": rid, "start": ss, "end": es,
                                    "start_idx": st, "end_idx": et})
        if st >= 0 and et >= 0:
            for i in range(st, et + 1): self._deleted_token_set.add(i)
            self._batch_apply_format(st, et, True)
        item = self._waveform.add_delete_region(rid, ss, es)
        item.sig_right_click.connect(lambda pos, r=rid: self._on_delete_region_right_click(r, pos))
        self._update_status()
        self._save_session()

    def _merge_into_vad(self, ss: float, es: float) -> None:
        # Expand range to absorb all overlapping regions (iterate until stable)
        changed = True
        while changed:
            changed = False
            for d in self._delete_ranges:
                if not (es <= d["start"] or ss >= d["end"]):
                    nss, nes = min(ss, d["start"]), max(es, d["end"])
                    if nss < ss or nes > es:
                        ss, es, changed = nss, nes, True
            for r in self._vad_regions:
                if not (es <= r["start"] or ss >= r["end"]):
                    nss, nes = min(ss, r["start"]), max(es, r["end"])
                    if nss < ss or nes > es:
                        ss, es, changed = nss, nes, True

        self._push_undo("add_vad", self._make_snapshot())

        del_ids = [d["id"] for d in self._delete_ranges
                   if not (es <= d["start"] or ss >= d["end"])]
        for did in del_ids:
            idx = next((i for i, d in enumerate(self._delete_ranges) if d["id"] == did), None)
            if idx is not None:
                self._remove_delete_range(idx)

        vad_ids = [r["id"] for r in self._vad_regions
                   if not (es <= r["start"] or ss >= r["end"])]
        for vid in vad_ids:
            self._remove_vad_region_by_id(vid)

        vid = self._next_vad_id; self._next_vad_id += 1
        item = self._waveform.add_vad_region(vid, ss, es)
        item.sig_right_click.connect(lambda pos, v=vid: self._on_vad_right_click(v, pos))
        item.sigRegionChangeFinished.connect(lambda _, v=vid: self._on_vad_boundary_changed(v))
        st, et = self._find_tokens_in_range(ss, es)
        self._vad_regions.append({"id": vid, "start": ss, "end": es,
                                  "start_tok": st, "end_tok": et})
        if st >= 0 and et >= 0:
            self._set_vad_tokens(st, et, True)
        self._refresh_all_extra_selections()
        self._update_status()
        self._save_session()
        self._vad_visible = True
        self._update_vad_btn_text()

    def _restore_segment_audio(self, ss: float, es: float) -> None:
        overlapping = [d["id"] for d in self._delete_ranges
                       if not (es <= d["start"] or ss >= d["end"])]
        if not overlapping: return
        self._push_undo("restore_delete", self._make_snapshot())
        for did in overlapping:
            idx = next((i for i, d in enumerate(self._delete_ranges) if d["id"] == did), None)
            if idx is not None:
                self._remove_delete_range(idx)
        self._save_session()

    def _restore_segment_vad(self, ss: float, es: float) -> None:
        overlapping = [r["id"] for r in self._vad_regions
                       if not (es <= r["start"] or ss >= r["end"])]
        if not overlapping: return
        self._push_undo("restore_vad", self._make_snapshot())
        for vid in overlapping:
            self._remove_vad_region_by_id(vid)
        self._update_status()
        self._save_session()

    # ── Status ──────────────────────────────────────────────────────────────────

    def _update_wpm_label(self) -> None:
        if not self._total_transcript_units:
            self._wpm_label.setText("")
            return
        dur = self._duration_s
        if dur <= 0:
            dur = self._waveform._duration_s
        if dur <= 0 and self._valid_tokens:
            dur = self._valid_tokens[-1]["end"]
        if dur <= 0:
            return
        silence = (sum(r["end"] - r["start"] for r in self._vad_regions)
                   if self._vad_regions else self._whisper_silence_s)
        speaking_s = max(dur - silence, 1.0)
        wpm = self._total_transcript_units / speaking_s * 60
        self._wpm_label.setText(f"語速: {wpm:.0f} 字/分")

    def _update_status(self) -> None:
        if self._exporting:
            return
        parts = [self._audio_name] if self._audio_name else []
        n_del = len(self._delete_ranges)
        if n_del:
            parts.append(f"{n_del} 段刪除 | 共 {sum(d['end']-d['start'] for d in self._delete_ranges):.1f} 秒")
        n_vad = len(self._vad_regions)
        if n_vad:
            parts.append(f"偵測到 {n_vad} 個非語音區間，共 {sum(r['end']-r['start'] for r in self._vad_regions):.1f} 秒")
        self._status_label.setText(
            "  |  ".join(parts) if parts else "請從 File > Open 開啟音檔")

    # ── Session save / restore ──────────────────────────────────────────────────

    @staticmethod
    def _fmt_ts(ts: str) -> str:
        try:
            return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return ts

    def _session_data(self) -> dict:
        return {
            "audio_path":    self._audio_path,
            "saved_at":      datetime.now().isoformat(timespec="seconds"),
            "delete_ranges": [{"id": d["id"], "start": d["start"], "end": d["end"]}
                               for d in self._delete_ranges],
            "vad_regions":   [{"id": r["id"], "start": r["start"], "end": r["end"]}
                               for r in self._vad_regions],
            "token_edits":   {str(k): v for k, v in self._token_edits.items()},
        }

    def _save_session(self) -> None:
        if not self._audio_path:
            return
        FILE_MANAGER.save_json(FILE_MANAGER.session_path, self._session_data())

    def _apply_session_data(self, data: dict) -> None:
        for dr in data.get("delete_ranges", []):
            ss, es = dr["start"], dr["end"]
            st, et = self._find_tokens_in_range(ss, es)
            rid    = dr["id"]
            self._delete_ranges.append({"id": rid, "start": ss, "end": es,
                                        "start_idx": st, "end_idx": et})
            if st >= 0 and et >= 0:
                for i in range(st, et + 1):
                    self._deleted_token_set.add(i)
                self._batch_apply_format(st, et, True)
            item = self._waveform.add_delete_region(rid, ss, es)
            item.sig_right_click.connect(
                lambda pos, r=rid: self._on_delete_region_right_click(r, pos))
            if rid >= self._next_range_id:
                self._next_range_id = rid + 1
        for vr in data.get("vad_regions", []):
            ss, es = vr["start"], vr["end"]
            vid    = vr["id"]
            item   = self._waveform.add_vad_region(vid, ss, es)
            item.sig_right_click.connect(
                lambda pos, v=vid: self._on_vad_right_click(v, pos))
            item.sigRegionChangeFinished.connect(
                lambda _, v=vid: self._on_vad_boundary_changed(v))
            st, et = self._find_tokens_in_range(ss, es)
            self._vad_regions.append({"id": vid, "start": ss, "end": es,
                                      "start_tok": st, "end_tok": et})
            self._set_vad_tokens(st, et, True)
            if vid >= self._next_vad_id:
                self._next_vad_id = vid + 1
        edits = data.get("token_edits", {})
        if edits and self._valid_tokens:
            for k, v in edits.items():
                try:
                    i = int(k)
                    if 0 <= i < len(self._valid_tokens):
                        self._valid_tokens[i]["text"] = v
                        self._token_edits[i] = v
                except (ValueError, IndexError):
                    pass
            if edits:
                self._refresh_transcript_display()
        if data.get("delete_ranges") or data.get("vad_regions"):
            self._update_status()
        self._update_vad_btn_text()

    def _restore_session_if_ready(self) -> None:
        if self._session_restored or not self._audio_path:
            return
        if not self._valid_tokens or not self._duration_s:
            return
        self._session_restored = True

        auto_data = FILE_MANAGER.load_json(FILE_MANAGER.session_path)
        ep_data   = FILE_MANAGER.load_json(
            FILE_MANAGER.get_episode_session_path(self._audio_path))

        # 自動存檔就是這集 → 直接還原，不提示
        if auto_data.get("audio_path") == self._audio_path:
            if auto_data.get("delete_ranges") or auto_data.get("vad_regions"):
                self._apply_session_data(auto_data)
            return

        # 自動存檔是別集 → 找手動存檔
        if not ep_data or (not ep_data.get("delete_ranges") and not ep_data.get("vad_regions")):
            return

        ep_ts   = ep_data.get("saved_at", "")
        auto_ts = auto_data.get("saved_at", "")
        ep_disp = self._fmt_ts(ep_ts)

        if ep_ts and auto_ts and ep_ts < auto_ts:
            auto_disp = self._fmt_ts(auto_ts)
            msg = (f"找到 {ep_disp} 的手動存檔，"
                   f"但比最後作業時間（{auto_disp}）早，是否仍要還原？")
        else:
            msg = f"找到 {ep_disp} 的手動存檔，是否還原？"

        if QMessageBox.question(self, "還原進度", msg) == QMessageBox.Yes:
            self._apply_session_data(ep_data)

    def _restore_from_manual_save(self) -> None:
        if not self._audio_path:
            QMessageBox.warning(self, "還原進度", "請先開啟音檔"); return
        ep_data = FILE_MANAGER.load_json(
            FILE_MANAGER.get_episode_session_path(self._audio_path))
        if not ep_data or (not ep_data.get("delete_ranges") and not ep_data.get("vad_regions")):
            QMessageBox.information(self, "還原進度", "沒有找到此音檔的手動存檔。"); return

        ep_ts   = ep_data.get("saved_at", "")
        ep_disp = self._fmt_ts(ep_ts)
        auto_data = FILE_MANAGER.load_json(FILE_MANAGER.session_path)
        auto_ts = (auto_data.get("saved_at", "")
                   if auto_data.get("audio_path") == self._audio_path else "")

        if ep_ts and auto_ts and ep_ts < auto_ts:
            auto_disp = self._fmt_ts(auto_ts)
            msg = (f"手動存檔時間：{ep_disp}\n目前進度時間：{auto_disp}\n"
                   f"手動存檔比目前進度早，還原後將覆蓋目前編輯，是否繼續？")
        else:
            msg = f"找到 {ep_disp} 的手動存檔，還原後將覆蓋目前編輯，是否繼續？"

        if QMessageBox.question(self, "還原進度", msg) == QMessageBox.Yes:
            self._push_undo("restore_manual", self._make_snapshot())
            self._clear_regions()
            self._apply_session_data(ep_data)
            self._save_session()

    def _try_restore_last_session(self) -> None:
        data = FILE_MANAGER.load_json(FILE_MANAGER.session_path)
        last_path = data.get("audio_path", "")
        if last_path and Path(last_path).exists():
            self._open_audio(last_path)

    # ── Keyboard / auto-scroll ──────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self._toggle_play(); event.accept(); return
        super().keyPressEvent(event)

    def _auto_scroll_waveform(self, pos_s: float) -> None:
        if not self._follow_mode:
            return
        dur = self._waveform._duration_s
        if not dur: return
        lo, hi = self._waveform.plotItem.vb.viewRange()[0]
        span = hi - lo
        if span >= dur * 0.99: return          # 全覽模式不自動捲動
        if pos_s > hi - span * 0.05 or pos_s < lo:
            new_lo = max(0.0, min(dur - span, pos_s - span * 0.2))
            self._waveform.setXRange(new_lo, new_lo + span, padding=0)

    def _on_follow_toggled(self, checked: bool):
        self._follow_mode = checked
        if checked and self._player.playbackState() == QMediaPlayer.PlayingState:
            pos_s = self._player.position() / 1000.0
            self._auto_scroll_waveform(pos_s)

    # ── Undo / Redo ──────────────────────────────────────────────────────────

    def _push_undo(self, action: str, snapshot: dict):
        self._undo_stack.append({"action": action, "snapshot": snapshot})
        self._redo_stack.clear()
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)

    def _make_snapshot(self) -> dict:
        return {
            "delete_ranges": [
                {"id": d["id"], "start": d["start"], "end": d["end"],
                 "start_idx": d["start_idx"], "end_idx": d["end_idx"]}
                for d in self._delete_ranges
            ],
            "vad_regions": [
                {"id": r["id"], "start": r["start"], "end": r["end"],
                 "start_tok": r["start_tok"], "end_tok": r["end_tok"]}
                for r in self._vad_regions
            ],
            "deleted_token_set": list(self._deleted_token_set),
            "vad_token_set": list(self._vad_token_set),
            "next_range_id": self._next_range_id,
            "next_vad_id": self._next_vad_id,
        }

    def _restore_snapshot(self, snap: dict):
        self._transcript_view.setUpdatesEnabled(False)
        self._highlighted_idx = -1
        # Reset all token formatting to default quickly
        if self._valid_tokens:
            fmt = QTextCharFormat()
            fmt.clearBackground(); fmt.clearForeground()
            cur = QTextCursor(self._transcript_view.document())
            cur.setPosition(0)
            cur.setPosition(self._transcript_view._char_ends[-1] if self._transcript_view._char_ends else 0,
                            QTextCursor.MoveMode.KeepAnchor)
            cur.setCharFormat(fmt)
        self._clear_regions()
        self._deleted_token_set = set(snap.get("deleted_token_set", []))
        self._vad_token_set = set(snap.get("vad_token_set", []))
        self._next_range_id = snap.get("next_range_id", 0)
        self._next_vad_id = snap.get("next_vad_id", 0)
        for dr in snap.get("delete_ranges", []):
            self._delete_ranges.append(dict(dr))
            self._batch_apply_format(dr["start_idx"], dr["end_idx"], True)
            item = self._waveform.add_delete_region(dr["id"], dr["start"], dr["end"])
            item.sig_right_click.connect(
                lambda pos, r=dr["id"]: self._on_delete_region_right_click(r, pos))
        for vr in snap.get("vad_regions", []):
            self._vad_regions.append(dict(vr))
            item = self._waveform.add_vad_region(vr["id"], vr["start"], vr["end"])
            item.sig_right_click.connect(
                lambda pos, v=vr["id"]: self._on_vad_right_click(v, pos))
            item.sigRegionChangeFinished.connect(
                lambda _, v=vr["id"]: self._on_vad_boundary_changed(v))
            self._set_vad_tokens(vr["start_tok"], vr["end_tok"], True)
        self._transcript_view.setUpdatesEnabled(True)
        self._update_status()
        self._save_session()
        self._update_vad_btn_text()
        self._update_filler_btn_text()

    def _undo(self):
        if not self._undo_stack:
            return
        snap = self._make_snapshot()
        state = self._undo_stack.pop()
        self._redo_stack.append({"action": state["action"], "snapshot": snap})
        self._restore_snapshot(state["snapshot"])

    def _redo(self):
        if not self._redo_stack:
            return
        snap = self._make_snapshot()
        state = self._redo_stack.pop()
        self._undo_stack.append({"action": state["action"], "snapshot": snap})
        self._restore_snapshot(state["snapshot"])

    def _check_skip_delete(self, pos_ms: int) -> None:
        for dr in self._delete_ranges:
            s, e = int(dr["start"] * 1000), int(dr["end"] * 1000)
            if s <= pos_ms < e:
                self._player.setPosition(e); return

    # ── Segment right-click ──────────────────────────────────────────────────────

    def _show_prompt_settings(self) -> None:
        cfg = FILE_MANAGER.load_config()
        text, ok = QInputDialog.getText(
            self, "設定提示詞",
            "用空格分隔詞彙，例如：消業障旅行團 Bryan 台灣\n節目名稱、人名等特定詞彙（可提升辨識率）：",
            text=cfg.get("initial_prompt", ""))
        if ok:
            cfg["initial_prompt"] = text.strip()
            FILE_MANAGER.save_config(cfg)

    def _show_segment_menu(self, seg_idx: int, gpos: QPoint) -> None:
        seg = self._segments[seg_idx]
        ss, es = float(seg["start"]), float(seg["end"])
        preview = seg["text"][:30] + ("…" if len(seg["text"]) > 30 else "")
        menu = QMenu(self)
        header = menu.addAction(f"§{seg_idx + 1}：{preview}")
        header.setEnabled(False)
        menu.addSeparator()

        has_del = any(not (es <= d["start"] or ss >= d["end"]) for d in self._delete_ranges)
        has_vad = any(not (es <= r["start"] or ss >= r["end"]) for r in self._vad_regions)

        if has_del:
            menu.addAction("恢復此音訊").triggered.connect(
                lambda: self._restore_segment_audio(ss, es))
            menu.addAction("改為靜音區間").triggered.connect(
                lambda: self._merge_into_vad(ss, es))
        if has_vad:
            menu.addAction("確認刪除此靜音區間").triggered.connect(
                lambda: self._merge_into_delete(ss, es))
            menu.addAction("移除此靜音標記").triggered.connect(
                lambda: self._restore_segment_vad(ss, es))
        if not has_del and not has_vad:
            menu.addAction("將此段標為刪除").triggered.connect(
                lambda: self._mark_segment_as_delete(seg_idx))
            menu.addAction("將此段標為靜音").triggered.connect(
                lambda: self._mark_segment_as_vad(seg_idx))

        menu.exec(gpos)

    def _on_waveform_right_click(self, t: float) -> None:
        if not self._segments: return
        seg_idx = next((i for i, s in enumerate(self._segments)
                        if float(s["start"]) <= t < float(s["end"])), -1)
        if seg_idx < 0:
            if t >= float(self._segments[-1]["end"]):
                seg_idx = len(self._segments) - 1
            else:
                return
        self._show_segment_menu(seg_idx, QCursor.pos())

    def _on_seg_bar_right_click(self, seg_idx: int, gpos: QPoint) -> None:
        if 0 <= seg_idx < len(self._segments):
            self._show_segment_menu(seg_idx, gpos)

    def _mark_segment_as_delete(self, seg_idx: int) -> None:
        seg = self._segments[seg_idx]
        ss, es = float(seg["start"]), float(seg["end"])
        self._merge_into_delete(ss, es)

    def _mark_segment_as_vad(self, seg_idx: int) -> None:
        seg = self._segments[seg_idx]
        ss, es = float(seg["start"]), float(seg["end"])
        self._merge_into_vad(ss, es)

    # ── Transcript text editing ──────────────────────────────────────────────────

    def _edit_token_text(self, idx: int) -> None:
        if not (0 <= idx < len(self._valid_tokens)): return
        seg_id = self._valid_tokens[idx].get("segment_id", -1)
        seg_indices = [i for i, t in enumerate(self._valid_tokens)
                       if t.get("segment_id", -1) == seg_id] if seg_id >= 0 else [idx]
        combined = "".join(self._valid_tokens[i]["text"] for i in seg_indices)
        new_text, ok = QInputDialog.getText(
            self, "編輯段落文字", "段落文字（可修改整段）：", text=combined)
        if not ok or new_text == combined: return
        first = seg_indices[0]
        self._valid_tokens[first]["text"] = new_text
        self._token_edits[first] = new_text
        for i in seg_indices[1:]:
            self._valid_tokens[i]["text"] = ""
            self._token_edits[i] = ""
        self._refresh_transcript_display()
        self._save_session()

    def _refresh_transcript_display(self) -> None:
        view = self._transcript_view
        pos = 0
        cs: list[int] = []
        ce: list[int] = []
        for tok in self._valid_tokens:
            cs.append(pos); pos += len(tok["text"]); ce.append(pos)

        view.setUpdatesEnabled(False)
        cur = QTextCursor(view.document())
        cur.select(QTextCursor.SelectionType.Document)
        cur.removeSelectedText()
        blk = QTextBlockFormat(); blk.setLineHeight(150, 1)
        cur.setBlockFormat(blk)
        cur.insertText("".join(t["text"] for t in self._valid_tokens))
        view.set_token_ranges(cs, ce)

        for dr in self._delete_ranges:
            st, et = dr.get("start_idx", -1), dr.get("end_idx", -1)
            if st >= 0 and et >= 0:
                self._batch_apply_format(st, et, True)
        if 0 <= self._highlighted_idx < len(self._valid_tokens):
            self._apply_single_format(self._highlighted_idx)
        self._refresh_all_extra_selections()
        view.setUpdatesEnabled(True)

    # ── Transcript export ────────────────────────────────────────────────────────

    @staticmethod
    def _srt_ts(s: float) -> str:
        ms = round(s * 1000)
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        sec, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

    def _export_txt(self) -> None:
        if not self._valid_tokens:
            QMessageBox.warning(self, "匯出", "尚未載入逐字稿"); return
        base = Path(self._audio_path).stem if self._audio_path else "transcript"
        default_dir = FILE_MANAGER.get_project_dir(self._audio_path) if self._audio_path else Path.cwd()
        def_path = str(default_dir / (base + "_transcript.txt"))
        out, _ = QFileDialog.getSaveFileName(
            self, "匯出逐字稿", def_path, "Text Files (*.txt);;All Files (*)")
        if not out:
            return
        lines = []
        current_seg_id = object()
        current: list[str] = []
        for i, t in enumerate(self._valid_tokens):
            if i in self._deleted_token_set:
                continue
            seg_id = t.get("segment_id", -1)
            if seg_id != current_seg_id:
                if current:
                    lines.append("".join(current).strip())
                current = []
                current_seg_id = seg_id
            current.append(t["text"])
        if current:
            lines.append("".join(current).strip())
        Path(out).write_text("\n".join(l for l in lines if l), encoding="utf-8")
        QMessageBox.information(self, "匯出完成", f"逐字稿已匯出：\n{out}")

    def _export_srt(self) -> None:
        if not self._valid_tokens:
            QMessageBox.warning(self, "匯出", "尚未載入逐字稿"); return
        base = Path(self._audio_path).stem if self._audio_path else "subtitle"
        default_dir = FILE_MANAGER.get_project_dir(self._audio_path) if self._audio_path else Path.cwd()
        def_path = str(default_dir / (base + "_subtitle.srt"))
        out, _ = QFileDialog.getSaveFileName(
            self, "匯出字幕", def_path, "SRT Files (*.srt);;All Files (*)")
        if not out:
            return
        entries: list[str] = []
        counter = 1
        current_seg_id = object()
        current: list[dict] = []

        def _flush():
            nonlocal counter
            if not current:
                return
            text = "".join(tt["text"] for tt in current).strip()
            if text:
                s = self._srt_ts(current[0]["start"])
                e = self._srt_ts(current[-1]["end"])
                entries.append(f"{counter}\n{s} --> {e}\n{text}\n")
                counter += 1
            current.clear()

        for i, t in enumerate(self._valid_tokens):
            if i in self._deleted_token_set:
                continue
            seg_id = t.get("segment_id", -1)
            if seg_id != current_seg_id:
                _flush()
                current_seg_id = seg_id
            current.append(t)
        _flush()

        Path(out).write_text("\n".join(entries), encoding="utf-8")
        QMessageBox.information(self, "匯出完成", f"字幕已匯出：\n{out}")

    # ── Export ───────────────────────────────────────────────────────────────────

    def _export(self) -> None:
        if not self._delete_ranges:
            QMessageBox.warning(self, "匯出", "尚未標記任何刪除區段"); return
        if self._export_thread and self._export_thread.isRunning():
            QMessageBox.information(self, "匯出", "匯出進行中，請稍候…"); return
        base_name = Path(self._audio_path).stem if self._audio_path else "exported"
        default_dir = FILE_MANAGER.get_project_dir(self._audio_path) if self._audio_path else Path.cwd()
        def_path = str(default_dir / (base_name + "_exported.m4a"))
        out, _ = QFileDialog.getSaveFileName(
            self, "匯出音檔", def_path,
            "M4A Files (*.m4a);;WAV Files (*.wav);;All Files (*)")
        if not out: return
        dur  = self._duration_s or self._player.duration() / 1000.0
        keep = compute_keep_ranges(dur, self._delete_ranges)
        if not keep:
            QMessageBox.warning(self, "匯出", "保留區段為空，無法匯出"); return
        self._export_act.setEnabled(False)
        self._exporting = True
        self._status_label.setText("匯出中…")
        t = QThread(); w = ExportWorker(self._audio_path, keep, out)
        w.moveToThread(t); t.started.connect(w.run)
        w.progress.connect(self._status_label.setText)
        w.elapsed.connect(self._on_export_elapsed)
        w.finished.connect(self._on_export_finished); w.error.connect(self._on_export_error)
        w.finished.connect(t.quit); w.error.connect(t.quit)
        t.finished.connect(self._on_export_thread_done)
        t.finished.connect(w.deleteLater); t.finished.connect(t.deleteLater)
        self._export_thread = t; self._export_worker = w; t.start()

    def _on_export_elapsed(self, secs: float):
        self._export_time_secs = secs

    def _on_export_finished(self, path: str) -> None:
        t = getattr(self, "_export_time_secs", 0)
        msg = f"匯出完成：{path}"
        if t:
            msg += f"\n花費時間：{int(t // 60)}分{int(t % 60)}秒" if t >= 60 else f"\n花費時間：{t:.1f}秒"
        QMessageBox.information(self, "匯出完成", msg)

    def _on_export_error(self, msg: str) -> None:
        QMessageBox.critical(self, "匯出失敗", msg)

    def _on_export_thread_done(self) -> None:
        self._exporting = False
        self._export_act.setEnabled(True); self._update_status()

    # ── Playback ─────────────────────────────────────────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "開啟音檔", "",
            "Audio Files (*.m4a *.wav *.mp4);;All Files (*)")
        if not path: return
        self._open_audio(path)

    def _open_audio(self, path: str) -> None:
        self._cancel_vad_if_running()
        resolved               = str(Path(path).resolve())
        self._audio_path       = resolved
        self._audio_name       = Path(resolved).name
        self._session_restored = False
        self._duration_s       = 0.0
        self._clear_regions(); self._clear_transcript()
        self._player.setSource(QUrl.fromLocalFile(resolved))
        self._start_waveform_worker(resolved)
        self._start_lufs_worker(resolved)
        self._play_btn.setText("播放")

        FILE_MANAGER.get_project_dir(resolved)

        self._save_session()

        tr_path = transcript_path_for(resolved)
        if tr_path.exists():
            self._status_label.setText("找到現有逐字稿，正在載入…")
            self._load_transcript_file(str(tr_path))
        else:
            self._status_label.setText("未找到逐字稿，請從 File > Transcribe 開始轉錄")

    def _load_transcript_file(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "載入失敗", str(e)); return
        self._populate_transcript(data)

    def _open_transcript_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "載入逐字稿", "",
            "JSON Files (*.json);;All Files (*)")
        if not path: return
        self._load_transcript_file(path)

    def _transcribe_audio(self):
        if not self._audio_path:
            QMessageBox.warning(self, "轉錄", "請先開啟音檔")
            return
        project_dir = FILE_MANAGER.get_project_dir(self._audio_path)
        tr_path = project_dir / "transcript.json"
        self._start_transcribe_with_dialog(self._audio_path, str(tr_path))

    def _save_episode_session(self) -> None:
        if not self._audio_path:
            QMessageBox.warning(self, "儲存進度", "尚未開啟音檔"); return
        ep_path = FILE_MANAGER.get_episode_session_path(self._audio_path)
        FILE_MANAGER.save_json(ep_path, self._session_data())
        QMessageBox.information(self, "儲存進度",
            f"已儲存至：{ep_path}\n時間：{self._fmt_ts(datetime.now().isoformat(timespec='seconds'))}")

    def _toggle_play(self):
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_token_clicked(self, idx: int):
        pos_ms = self._token_starts_ms[idx]
        self._player.setPosition(pos_ms)
        pos_s = pos_ms / 1000.0
        dur = self._waveform._duration_s
        if dur > 0:
            lo, hi = self._waveform.plotItem.vb.viewRange()[0]
            span = hi - lo
            if span < dur * 0.99 and not (lo + span * 0.05 <= pos_s <= hi - span * 0.05):
                new_lo = max(0.0, min(dur - span, pos_s - span * 0.3))
                self._waveform.setXRange(new_lo, new_lo + span, padding=0)

    def _on_scrubber_pressed(self):   self._seeking = True
    def _on_scrubber_released(self):
        self._player.setPosition(self._scrubber.value()); self._seeking = False

    def _on_scrubber_value_changed(self, v: int):
        if self._seeking: self._update_time_label(v, self._player.duration())

    def _on_duration_changed(self, dur_ms: int):
        self._scrubber.setRange(0, dur_ms)
        self._duration_s = dur_ms / 1000.0
        self._vad_btn.setEnabled(True)
        self._update_time_label(self._player.position(), dur_ms)
        self._update_wpm_label()
        self._restore_session_if_ready()

    def _on_position_changed(self, pos_ms: int):
        if not self._seeking: self._scrubber.setValue(pos_ms)
        self._update_time_label(pos_ms, self._player.duration())
        self._update_highlight(pos_ms)
        pos_s = pos_ms / 1000.0
        self._waveform.set_playhead(pos_s)
        if not self._seeking and self._player.playbackState() == QMediaPlayer.PlayingState:
            self._auto_scroll_waveform(pos_s)
            self._check_skip_delete(pos_ms)

    def _on_state_changed(self, state):
        self._play_btn.setText("暫停" if state == QMediaPlayer.PlayingState else "播放")

    def _update_time_label(self, pos: int, dur: int):
        self._time_label.setText(f"{fmt_time(pos)} / {fmt_time(dur)}")

    def _update_highlight(self, pos_ms: int) -> None:
        if not self._token_starts_ms: return
        idx = bisect.bisect_right(self._token_starts_ms, pos_ms) - 1
        if idx >= 0 and pos_ms >= int(self._valid_tokens[idx]["end"] * 1000):
            idx = -1
        if idx == self._highlighted_idx: return
        prev, self._highlighted_idx = self._highlighted_idx, idx
        if prev >= 0: self._apply_single_format(prev)
        if idx  >= 0: self._apply_single_format(idx); self._scroll_to_token(idx)

    def _scroll_to_token(self, idx: int) -> None:
        view = self._transcript_view
        cur  = QTextCursor(view.document())
        cur.setPosition(view._char_starts[idx])
        rect = view.cursorRect(cur)
        vbar = view.verticalScrollBar()
        vbar.setValue(max(0, vbar.value() + rect.top() - view.viewport().height() // 2))


def main():
    app = QApplication(sys.argv)
    icon_path = Path(__file__).parent / "icon.icns"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
