from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler


FEATURE_GROUPS = ["header", "section", "imports", "strings", "all"]

MACHINE_TYPES = [
    "IMAGE_FILE_MACHINE_UNKNOWN", "IMAGE_FILE_MACHINE_I386", "IMAGE_FILE_MACHINE_R3000", "IMAGE_FILE_MACHINE_R4000",
    "IMAGE_FILE_MACHINE_R10000", "IMAGE_FILE_MACHINE_WCEMIPSV2", "IMAGE_FILE_MACHINE_ALPHA", "IMAGE_FILE_MACHINE_SH3",
    "IMAGE_FILE_MACHINE_SH3DSP", "IMAGE_FILE_MACHINE_SH3E", "IMAGE_FILE_MACHINE_SH4", "IMAGE_FILE_MACHINE_SH5",
    "IMAGE_FILE_MACHINE_ARM", "IMAGE_FILE_MACHINE_THUMB", "IMAGE_FILE_MACHINE_ARMNT", "IMAGE_FILE_MACHINE_AM33",
    "IMAGE_FILE_MACHINE_POWERPC", "IMAGE_FILE_MACHINE_POWERPCFP", "IMAGE_FILE_MACHINE_IA64", "IMAGE_FILE_MACHINE_MIPS16",
    "IMAGE_FILE_MACHINE_ALPHA64", "IMAGE_FILE_MACHINE_AXP64", "IMAGE_FILE_MACHINE_MIPSFPU", "IMAGE_FILE_MACHINE_MIPSFPU16",
    "IMAGE_FILE_MACHINE_TRICORE", "IMAGE_FILE_MACHINE_CEF", "IMAGE_FILE_MACHINE_EBC", "IMAGE_FILE_MACHINE_RISCV32",
    "IMAGE_FILE_MACHINE_RISCV64", "IMAGE_FILE_MACHINE_RISCV128", "IMAGE_FILE_MACHINE_LOONGARCH32", "IMAGE_FILE_MACHINE_LOONGARCH64",
    "IMAGE_FILE_MACHINE_AMD64", "IMAGE_FILE_MACHINE_M32R", "IMAGE_FILE_MACHINE_ARM64", "IMAGE_FILE_MACHINE_CEE",
]
MACHINE_TYPES_DICT = {name: idx for idx, name in enumerate(MACHINE_TYPES)}

SUBSYSTEM_TYPES = [
    "IMAGE_SUBSYSTEM_UNKNOWN", "IMAGE_SUBSYSTEM_NATIVE", "IMAGE_SUBSYSTEM_WINDOWS_GUI", "IMAGE_SUBSYSTEM_WINDOWS_CUI",
    "IMAGE_SUBSYSTEM_OS2_CUI", "IMAGE_SUBSYSTEM_POSIX_CUI", "IMAGE_SUBSYSTEM_NATIVE_WINDOWS", "IMAGE_SUBSYSTEM_WINDOWS_CE_GUI",
    "IMAGE_SUBSYSTEM_EFI_APPLICATION", "IMAGE_SUBSYSTEM_EFI_BOOT_SERVICE_DRIVER", "IMAGE_SUBSYSTEM_EFI_RUNTIME_DRIVER",
    "IMAGE_SUBSYSTEM_EFI_ROM", "IMAGE_SUBSYSTEM_XBOX", "IMAGE_SUBSYSTEM_WINDOWS_BOOT_APPLICATION",
]
SUBSYSTEM_TYPES_DICT = {name: idx for idx, name in enumerate(SUBSYSTEM_TYPES)}

IMAGE_CHARACTERISTICS = [
    "RELOCS_STRIPPED", "EXECUTABLE_IMAGE", "LINE_NUMS_STRIPPED", "LOCAL_SYMS_STRIPPED", "AGGRESIVE_WS_TRIM",
    "LARGE_ADDRESS_AWARE", "16BIT_MACHINE", "BYTES_REVERSED_LO", "32BIT_MACHINE", "DEBUG_STRIPPED",
    "REMOVABLE_RUN_FROM_SWAP", "NET_RUN_FROM_SWAP", "SYSTEM", "DLL", "UP_SYSTEM_ONLY", "BYTES_REVERSED_HI",
]

DLL_CHARACTERISTICS = [
    "HIGH_ENTROPY_VA", "DYNAMIC_BASE", "FORCE_INTEGRITY", "NX_COMPAT", "NO_ISOLATION",
    "NO_SEH", "NO_BIND", "APPCONTAINER", "WDM_DRIVER", "GUARD_CF", "TERMINAL_SERVER_AWARE",
]

DOS_MEMBERS = [
    "e_magic", "e_cblp", "e_cp", "e_crlc", "e_cparhdr", "e_minalloc", "e_maxalloc", "e_ss", "e_sp",
    "e_csum", "e_ip", "e_cs", "e_lfarlc", "e_ovno", "e_oemid", "e_oeminfo", "e_lfanew",
]

STRING_REGEX_KEYS = [
    ".click(", "/EmbeddedFile", "/FlateDecode", "/URI", "/bin/", "/dev/", "/proc/", "/tmp/", "/usr/",
    "<script", "Invoke-Command", "Invoke-Expression", "Start-process", "base64", "base64string", "btc_wallet",
    "cache", "certificate", "clipboard", "command", "connect", "cookie", "create", "crypt", "debug", "decode",
    "delete", "desktop", "directory", "disk", "dos_msg", "download", "email_addr", "encode", "enum",
    "environment", "exit", "file", "file_path", "ftp", "get", "hidden", "hostname", "html", "http", "http://",
    "https://", "install", "internet", "ipv4_addr", "ipv6_addr", "javascript", "keyboard", "mac_addr", "memory",
    "module", "mutex", "onlick", "password", "post", "powershell", "privilege", "process", "registry_key",
    "remote", "resource", "security", "service", "shell", "snapshot", "system", "thread", "token", "url",
    "useragent", "wallet", "window",
]


@dataclass(frozen=True)
class Sample:
    sha256: str
    label: int
    week_id: str
    source_path: str
    header_vec: np.ndarray
    section_vec: np.ndarray
    imports_vec: np.ndarray
    strings_vec: np.ndarray

    @property
    def all_vec(self) -> np.ndarray:
        return np.concatenate(
            [self.header_vec, self.section_vec, self.imports_vec, self.strings_vec]
        ).astype(np.float32, copy=False)


def vectorize_header(raw_obj: dict) -> np.ndarray:
    if not raw_obj:
        return np.zeros(74, dtype=np.float32)
    coff = raw_obj["coff"]
    optional = raw_obj["optional"]
    dos = raw_obj["dos"]
    return np.hstack(
        [
            coff["timestamp"],
            coff["number_of_sections"],
            coff["number_of_symbols"],
            coff["sizeof_optional_header"],
            coff["pointer_to_symbol_table"],
            MACHINE_TYPES_DICT.get(coff["machine"], 0),
            SUBSYSTEM_TYPES_DICT.get(optional["subsystem"], 0),
            optional["major_image_version"],
            optional["minor_image_version"],
            optional["major_linker_version"],
            optional["minor_linker_version"],
            optional["major_operating_system_version"],
            optional["minor_operating_system_version"],
            optional["major_subsystem_version"],
            optional["minor_subsystem_version"],
            optional["sizeof_code"],
            optional["sizeof_headers"],
            optional["sizeof_image"],
            optional["sizeof_initialized_data"],
            optional["sizeof_uninitialized_data"],
            optional["sizeof_stack_reserve"],
            optional["sizeof_stack_commit"],
            optional["sizeof_heap_reserve"],
            optional["sizeof_heap_commit"],
            optional["address_of_entrypoint"],
            optional["base_of_code"],
            optional["image_base"],
            optional["section_alignment"],
            optional["checksum"],
            optional["number_of_rvas_and_sizes"],
            [ch in coff["characteristics"] for ch in IMAGE_CHARACTERISTICS],
            [ch in optional["dll_characteristics"] for ch in DLL_CHARACTERISTICS],
            [dos[member] for member in DOS_MEMBERS],
        ]
    ).astype(np.float32)


def vectorize_section(raw_obj: dict) -> np.ndarray:
    if not raw_obj:
        return np.zeros(224, dtype=np.float32)
    sections = raw_obj["sections"]
    n_sections = len(sections)
    n_zero_size = sum(1 for s in sections if s["size"] == 0)
    n_empty_name = sum(1 for s in sections if s["name"] == "")
    n_rx = sum(1 for s in sections if "MEM_READ" in s["props"] and "MEM_EXECUTE" in s["props"])
    n_w = sum(1 for s in sections if "MEM_WRITE" in s["props"])
    entropies = [s["entropy"] for s in sections] + [raw_obj["overlay"]["entropy"]] + [0]
    size_ratios = [s["size_ratio"] for s in sections] + [raw_obj["overlay"]["size_ratio"]] + [0]
    vsize_ratios = [s["vsize_ratio"] for s in sections] + [0]
    general = [
        n_sections, n_zero_size, n_empty_name, n_rx, n_w,
        max(entropies), min(entropies), max(size_ratios), min(size_ratios), max(vsize_ratios), min(vsize_ratios),
    ]
    section_sizes = [(s["name"], s["size"]) for s in sections]
    section_vsize = [(s["name"], s["vsize"]) for s in sections]
    section_entropy = [(s["name"], s["entropy"]) for s in sections]
    characteristics = [f"{s['name']}:{prop}" for s in sections for prop in s["props"]]
    section_sizes_hashed = FeatureHasher(50, input_type="pair").transform([section_sizes]).toarray()[0]
    section_vsize_hashed = FeatureHasher(50, input_type="pair").transform([section_vsize]).toarray()[0]
    section_entropy_hashed = FeatureHasher(50, input_type="pair").transform([section_entropy]).toarray()[0]
    characteristics_hashed = FeatureHasher(50, input_type="string").transform([characteristics]).toarray()[0]
    entry_name_hashed = FeatureHasher(10, input_type="string").transform([[raw_obj["entry"]]]).toarray()[0]
    return np.hstack(
        [
            general,
            section_sizes_hashed,
            section_vsize_hashed,
            section_entropy_hashed,
            characteristics_hashed,
            entry_name_hashed,
            raw_obj["overlay"]["size"],
            raw_obj["overlay"]["size_ratio"],
            raw_obj["overlay"]["entropy"],
        ]
    ).astype(np.float32)


def vectorize_imports(raw_obj: dict) -> np.ndarray:
    if not raw_obj:
        return np.zeros(1282, dtype=np.float32)
    libraries = list({lib.lower() for lib in raw_obj.keys()})
    libraries_hashed = FeatureHasher(256, input_type="string", alternate_sign=False).transform([libraries]).toarray()[0]
    imports = [lib.lower() + ":" + entry for lib, items in raw_obj.items() for entry in items]
    imports_hashed = FeatureHasher(1024, input_type="string", alternate_sign=False).transform([imports]).toarray()[0]
    lengths = [len(imports), len(libraries)]
    return np.hstack([lengths, libraries_hashed, imports_hashed]).astype(np.float32)


def vectorize_strings(raw_obj: dict) -> np.ndarray:
    if not raw_obj:
        return np.zeros(177, dtype=np.float32)
    regex_idxs = {key: idx for idx, key in enumerate(STRING_REGEX_KEYS)}
    hist_divisor = float(raw_obj["printables"]) if raw_obj.get("printables", 0) > 0 else 1.0
    string_counts = np.zeros(len(regex_idxs), dtype=np.float32)
    for regex, count in (raw_obj.get("string_counts") or {}).items():
        idx = regex_idxs.get(regex)
        if idx is not None:
            string_counts[idx] = count
    return np.hstack(
        [
            raw_obj.get("numstrings", 0),
            raw_obj.get("avlength", 0.0),
            raw_obj.get("printables", 0),
            np.asarray(raw_obj.get("printabledist", [0.0] * 96), dtype=np.float32) / hist_divisor,
            raw_obj.get("entropy", 0.0),
            string_counts,
        ]
    ).astype(np.float32)


def extract_feature_container(row: dict) -> dict:
    raw = row.get("raw")
    if isinstance(raw, dict) and raw:
        return raw
    return row


def build_sample(row: dict, source_path: Path) -> Sample | None:
    label = row.get("label")
    sha256 = row.get("sha256")
    if label not in {0, 1} or not sha256:
        return None
    raw = extract_feature_container(row)
    return Sample(
        sha256=str(sha256),
        label=int(label),
        week_id=str(row.get("week_id") or source_path.stem),
        source_path=str(source_path),
        header_vec=vectorize_header(raw.get("header") or {}),
        section_vec=vectorize_section(raw.get("section") or {}),
        imports_vec=vectorize_imports(raw.get("imports") or {}),
        strings_vec=vectorize_strings(raw.get("strings") or {}),
    )


def is_packed_row(row: dict) -> bool:
    raw = extract_feature_container(row)
    packer = raw.get("packer")
    if isinstance(packer, list):
        return len(packer) > 0
    if isinstance(packer, str):
        return bool(packer.strip())
    return bool(packer)


def matches_packer_filter(row: dict, packer_filter: str) -> bool:
    if packer_filter == "all":
        return True
    packed = is_packed_row(row)
    if packer_filter == "packed":
        return packed
    if packer_filter == "unpacked":
        return not packed
    raise ValueError(f"Unknown packer_filter: {packer_filter}")


def load_samples_from_jsonl(path: Path, packer_filter: str = "all") -> List[Sample]:
    samples: List[Sample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not matches_packer_filter(row, packer_filter):
                continue
            sample = build_sample(row, path)
            if sample is not None:
                samples.append(sample)
    return samples


def load_samples_from_paths(paths: Sequence[Path], packer_filter: str = "all") -> List[Sample]:
    samples: List[Sample] = []
    for path in paths:
        samples.extend(load_samples_from_jsonl(path, packer_filter=packer_filter))
    return samples


def balance_samples(
    samples: Sequence[Sample],
    seed: int,
    max_per_class: int | None = None,
) -> List[Sample]:
    benign = [sample for sample in samples if sample.label == 0]
    malware = [sample for sample in samples if sample.label == 1]
    target = min(len(benign), len(malware))
    if max_per_class is not None:
        target = min(target, max_per_class)
    rng = np.random.default_rng(seed)

    def pick(rows: List[Sample]) -> List[Sample]:
        if len(rows) <= target:
            return list(rows)
        idx = rng.choice(len(rows), size=target, replace=False)
        return [rows[i] for i in idx]

    balanced = pick(benign) + pick(malware)
    rng.shuffle(balanced)
    return balanced


def select_feature_group(samples: Sequence[Sample], group: str) -> np.ndarray:
    if group == "header":
        return np.stack([s.header_vec for s in samples]).astype(np.float64)
    if group == "section":
        return np.stack([s.section_vec for s in samples]).astype(np.float64)
    if group == "imports":
        return np.stack([s.imports_vec for s in samples]).astype(np.float64)
    if group == "strings":
        return np.stack([s.strings_vec for s in samples]).astype(np.float64)
    if group == "all":
        return np.stack([s.all_vec for s in samples]).astype(np.float64)
    raise ValueError(f"Unknown feature group: {group}")


def labels_array(samples: Sequence[Sample]) -> np.ndarray:
    return np.asarray([sample.label for sample in samples], dtype=int)


def stabilize_features(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e12, 1e12)
    # Compress extreme magnitudes while preserving sign so linear models remain numerically stable.
    return np.sign(X) * np.log1p(np.abs(X))


def scale_train_test(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_train = stabilize_features(X_train)
    X_test = stabilize_features(X_test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = np.clip(np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)
    X_test_scaled = np.clip(np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)
    return X_train_scaled, X_test_scaled


def summarize(values: Sequence[float]) -> dict:
    arr = np.asarray(list(values), dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def feature_rankings(score_map: Dict[str, float], higher_is_better: bool = True) -> Dict[str, int]:
    if higher_is_better:
        ordered = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))
    else:
        ordered = sorted(score_map.items(), key=lambda item: (item[1], item[0]))
    return {group: idx + 1 for idx, (group, _) in enumerate(ordered)}


def week_paths(data_root: Path, platform: str, weeks: Iterable[str], split: str) -> List[Path]:
    return [data_root / f"{week}_{platform}_{split}.jsonl" for week in weeks]
