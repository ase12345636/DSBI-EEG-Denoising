"""Download every public asset into its task-owned ``dataset/.../data`` folder.

The two Kaggle sources require a Kaggle account.  Put ``kaggle.json`` in the
standard location or set KAGGLE_USERNAME/KAGGLE_KEY before running main.py.
CHB-MIT is downloaded anonymously and resumably. The IC-U-Net checkpoint is bundled with the project and checksum-verified before use.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from utils.progress import progress


PHYSIONET_BASE = "https://physionet.org/files/chbmit/1.0.0"
ICUNET_CHECKPOINT_SHA256 = (
    "ab1034e921651f93a7b2617b6d28f77bf299a8f34a71b3841b36e84e3699a47c"
)


class DownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetSource:
    name: str
    provider: str
    identifier: str
    destination: Path
    expected: tuple[str, ...]


def sources(root: Path) -> dict[str, DatasetSource]:
    return {
        "bci_errp": DatasetSource(
            name="BCI Challenge @ NER 2015",
            provider="kaggle_competition",
            identifier="inria-bci-challenge",
            destination=root / "dataset" / "bci_errp" / "data" / "raw",
            expected=("TrainLabels.csv", "Data_S02_Sess01.csv"),
        ),
        "seizure_detection": DatasetSource(
            name="CHB-MIT recordings with seizures",
            provider="physionet_chbmit_seizures",
            identifier=f"{PHYSIONET_BASE}/RECORDS-WITH-SEIZURES",
            destination=root / "dataset" / "seizure_detection" / "data" / "raw",
            expected=("RECORDS-WITH-SEIZURES", "chb01-summary.txt"),
        ),
        "attention_state": DatasetSource(
            name="EEG Data for Mental Attention State Detection",
            provider="kaggle_dataset",
            identifier="inancigdem/eeg-data-for-mental-attention-state-detection",
            destination=root / "dataset" / "attention_state" / "data" / "raw",
            expected=("*.mat",),
        ),
    }


def describe_source(source: DatasetSource) -> str:
    auth = "Kaggle credentials + accepted rules" if source.provider.startswith("kaggle") else "anonymous"
    return f"{source.name}: {source.provider} ({auth}) -> {source.destination}"


_ATTENTION_RECORD_RE = re.compile(
    r"^eeg_record(\d+)\.mat$",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_attention_release(destination: Path) -> Path:
    """Return one canonical directory containing eeg_record1..34.mat.

    The Kaggle archive can contain two identical directory trees.  Raw data are
    left untouched; the pipeline is simply pointed at one verified complete
    copy so the Attention task sees 34 files instead of 68.
    """
    destination = Path(destination).resolve()
    by_parent: dict[Path, dict[int, Path]] = {}

    for path in destination.rglob("*.mat"):
        if not path.is_file():
            continue
        match = _ATTENTION_RECORD_RE.fullmatch(path.name)
        if match is None:
            continue
        number = int(match.group(1))
        if not 1 <= number <= 34:
            continue
        parent_files = by_parent.setdefault(path.parent.resolve(), {})
        if number in parent_files:
            raise DownloadError(
                "Attention directory contains duplicate recording numbers in "
                f"the same folder: {path.parent}"
            )
        parent_files[number] = path.resolve()

    expected = set(range(1, 35))
    complete = [
        (parent, files)
        for parent, files in by_parent.items()
        if set(files) == expected
    ]

    if not complete:
        found = sum(len(files) for files in by_parent.values())
        raise DownloadError(
            "Could not find one Attention directory containing exactly "
            f"eeg_record1.mat through eeg_record34.mat under {destination}; "
            f"found {found} matching files."
        )

    complete.sort(
        key=lambda item: (
            len(item[0].relative_to(destination).parts),
            str(item[0].relative_to(destination)).casefold(),
        )
    )
    selected_parent, selected_files = complete[0]

    # If multiple complete copies exist, verify that they are identical before
    # deterministically selecting one.  This protects the benchmark from
    # silently choosing conflicting data while still leaving the files intact.
    for other_parent, other_files in complete[1:]:
        for number in range(1, 35):
            selected = selected_files[number]
            other = other_files[number]
            if selected.stat().st_size != other.stat().st_size:
                raise DownloadError(
                    f"Conflicting Attention copies for eeg_record{number}.mat: "
                    f"{selected} and {other} have different sizes."
                )
            if _sha256(selected) != _sha256(other):
                raise DownloadError(
                    f"Conflicting Attention copies for eeg_record{number}.mat: "
                    f"{selected} and {other} have different contents."
                )

    total_matching = sum(len(files) for files in by_parent.values())
    if len(complete) > 1:
        print(
            "[download] Attention contains "
            f"{len(complete)} verified identical 34-file copies "
            f"({total_matching} matching files total); using {selected_parent}"
        )
    else:
        print(f"[download] Attention data directory: {selected_parent}")

    return selected_parent


def resolve_dataset_path(root: Path, task_name: str) -> Path:
    """Return the directory that should be passed to a task's prepare()."""
    try:
        source = sources(root)[task_name]
    except KeyError as exc:
        raise DownloadError(
            f"No downloader registered for task {task_name!r}"
        ) from exc

    if task_name == "attention_state":
        return _resolve_attention_release(source.destination)
    return source.destination


def ensure_dataset(root: Path, task_name: str, force: bool = False) -> Path:
    try:
        source = sources(root)[task_name]
    except KeyError as exc:
        raise DownloadError(f"No downloader registered for task {task_name!r}") from exc

    marker = source.destination / ".download-complete.json"
    if not force and marker.exists() and _expected_files_exist(source):
        print(f"[download] reuse {source.name}: {source.destination}")
        return resolve_dataset_path(root, task_name)

    source.destination.mkdir(parents=True, exist_ok=True)
    _check_free_space(source.destination)
    print(f"[download] {describe_source(source)}")
    if source.provider == "kaggle_competition":
        _kaggle_download("competitions", source.identifier, source.destination)
    elif source.provider == "kaggle_dataset":
        _kaggle_download("datasets", source.identifier, source.destination)
    elif source.provider == "physionet_chbmit_seizures":
        _download_chbmit_seizure_recordings(source.destination)
    else:
        raise DownloadError(f"Unsupported provider: {source.provider}")

    if not _expected_files_exist(source):
        raise DownloadError(
            f"Download finished but expected files were not found in {source.destination}: {source.expected}"
        )
    marker.write_text(
        json.dumps({"name": source.name, "provider": source.provider, "identifier": source.identifier}, indent=2),
        encoding="utf-8",
    )
    return resolve_dataset_path(root, task_name)


def ensure_icunet_checkpoint(root: Path, force: bool = False) -> Path:
    """Return the bundled IC-U-Net checkpoint after checksum verification."""
    del force
    destination = (
        root / "denoise" / "ic_unet" / "weights" / "BEST_checkpoint.pth.tar"
    )
    if not destination.exists():
        raise DownloadError(
            "The bundled IC-U-Net checkpoint is missing: "
            f"{destination}. Restore BEST_checkpoint.pth.tar before running IC-U-Net."
        )
    if destination.stat().st_size < 1_000_000:
        raise DownloadError("The bundled IC-U-Net checkpoint is unexpectedly small")
    digest = _sha256(destination)
    if digest != ICUNET_CHECKPOINT_SHA256:
        raise DownloadError(
            "IC-U-Net checkpoint checksum mismatch: "
            f"expected {ICUNET_CHECKPOINT_SHA256}, got {digest}"
        )
    return destination


def _kaggle_download(kind: str, identifier: str, destination: Path) -> None:
    if not _has_kaggle_credentials():
        raise DownloadError(
            "Kaggle credentials are missing. Create ~/.kaggle/kaggle.json (chmod 600), or set "
            "KAGGLE_USERNAME and KAGGLE_KEY. For BCI, also accept the competition rules in a browser."
        )
    try:
        import kaggle  # noqa: F401
    except ImportError as exc:
        raise DownloadError("Install dependencies first: python -m pip install -r requirements.txt") from exc

    if kind == "competitions":
        command = [
            sys.executable,
            "-m",
            "kaggle.cli",
            "competitions",
            "download",
            "-c",
            identifier,
            "-p",
            str(destination),
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "kaggle.cli",
            "datasets",
            "download",
            "-d",
            identifier,
            "-p",
            str(destination),
        ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise DownloadError(
            f"Kaggle download failed for {identifier}. Check credentials and dataset/competition access."
        ) from exc
    _extract_all_zips(destination)


def _download_chbmit_seizure_recordings(destination: Path) -> None:
    records_file = destination / "RECORDS-WITH-SEIZURES"
    _download_url(f"{PHYSIONET_BASE}/RECORDS-WITH-SEIZURES", records_file)
    records = [line.strip() for line in records_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise DownloadError("PhysioNet returned an empty RECORDS-WITH-SEIZURES file")

    patients = sorted({record.split("/", 1)[0] for record in records})
    jobs: list[tuple[str, Path, str]] = []

    for patient in patients:
        summary = destination / patient / f"{patient}-summary.txt"
        jobs.append((
            f"{PHYSIONET_BASE}/{patient}/{patient}-summary.txt",
            summary,
            summary.name,
        ))

    for record in records:
        jobs.append((
            f"{PHYSIONET_BASE}/{record}",
            destination / record,
            record,
        ))

    workers = 8

    pending = [job for job in jobs if not _download_is_complete(job[1])]
    already_done = len(jobs) - len(pending)
    print(
        f"[download] CHB-MIT parallel mode: workers={workers}, "
        f"pending={len(pending)}, reuse={already_done}"
    )

    with progress(
        total=len(jobs),
        initial=already_done,
        desc="CHB-MIT files",
        unit="file",
    ) as file_bar:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_label = {
                executor.submit(
                    _download_url,
                    url,
                    target,
                    show_progress=False,
                ): label
                for url, target, label in pending
            }
            for future in as_completed(future_to_label):
                label = future_to_label[future]
                file_bar.set_postfix_str(label, refresh=False)
                future.result()
                file_bar.update()


def _download_is_complete(destination: Path) -> bool:
    return destination.is_file() and destination.stat().st_size > 0


def _download_url(
    url: str,
    destination: Path,
    *,
    show_progress: bool = True,
    retries: int = 5,
    skip_existing: bool = True,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    # A completed file has already been atomically renamed from .part.
    # Reuse it when an interrupted dataset run is restarted.
    if skip_existing and _download_is_complete(destination):
        return

    partial = destination.with_name(destination.name + ".part")
    last_error: BaseException | None = None

    for attempt in range(1, retries + 1):
        headers: dict[str, str] = {"User-Agent": "eeg-denoising-benchmark/2.0"}
        existing = partial.stat().st_size if partial.exists() else 0
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                append = existing > 0 and getattr(response, "status", None) == 206
                mode = "ab" if append else "wb"
                content_length = int(response.headers.get("Content-Length", 0))
                total = existing + content_length if append else content_length
                initial = existing if append else 0

                with partial.open(mode) as handle:
                    if show_progress:
                        with progress(
                            total=total or None,
                            initial=initial,
                            desc=f"Download {destination.name}",
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            leave=False,
                        ) as byte_bar:
                            while True:
                                chunk = response.read(1024 * 1024)
                                if not chunk:
                                    break
                                handle.write(chunk)
                                byte_bar.update(len(chunk))
                    else:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)

            partial.replace(destination)
            return

        except (
            urllib.error.URLError,
            http.client.IncompleteRead,
            TimeoutError,
            ConnectionError,
            OSError,
        ) as exc:
            last_error = exc
            if attempt == retries:
                break
            # Keep the .part file and resume it on the next attempt.
            time.sleep(min(2 ** attempt, 30))

    raise DownloadError(
        f"Could not download {url} after {retries} attempts; "
        "partial file kept for resume"
    ) from last_error


def _extract_all_zips(destination: Path) -> None:
    extracted: set[Path] = set()
    while True:
        archives = [
            archive for archive in sorted(destination.rglob("*.zip"))
            if archive.resolve() not in extracted
        ]
        if not archives:
            return
        for archive in archives:
            _safe_extract_zip(archive, destination)
            extracted.add(archive.resolve())


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        for member in members:
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise DownloadError(f"Unsafe path in archive {archive}: {member.filename}")
        with progress(
            total=sum(member.file_size for member in members),
            desc=f"Extract {archive.name}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
        ) as extract_bar:
            for member in members:
                handle.extract(member, destination)
                extract_bar.update(member.file_size)


def _has_kaggle_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    configured = os.environ.get("KAGGLE_CONFIG_DIR")
    locations = [Path(configured) / "kaggle.json"] if configured else []
    locations.append(Path.home() / ".kaggle" / "kaggle.json")
    return any(path.is_file() for path in locations)


def _expected_files_exist(source: DatasetSource) -> bool:
    for pattern in source.expected:
        if not any(source.destination.rglob(pattern)):
            return False
    return True


def _check_free_space(destination: Path) -> None:
    free_gib = shutil.disk_usage(destination).free / (1024**3)
    if free_gib < 10:
        raise DownloadError(f"Only {free_gib:.1f} GiB free at {destination}; at least 10 GiB is required")
