"""
Batched, faster re-implementations of DCSA_Open_Science/artifacts/shared/common.py's
vectorize_header / vectorize_imports, used ONLY to speed up bulk data loading for this
supplementary re-analysis. Correctness is verified against the original per-row functions
in `selftest_against_common.py` before use on the full dataset -- DCSA_Open_Science itself
is never modified, only imported as a read-only library.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.feature_extraction import FeatureHasher

DCSA_SHARED = Path(__file__).resolve().parents[4] / "artifacts/shared"
sys.path.insert(0, str(DCSA_SHARED))

from common import (  # type: ignore  # noqa: E402
    MACHINE_TYPES_DICT,
    SUBSYSTEM_TYPES_DICT,
    IMAGE_CHARACTERISTICS,
    DLL_CHARACTERISTICS,
    DOS_MEMBERS,
)

HEADER_DIM = 74
IMPORTS_DIM = 1282

_LIB_HASHER = FeatureHasher(256, input_type="string", alternate_sign=False)
_IMPORT_HASHER = FeatureHasher(1024, input_type="string", alternate_sign=False)


def vectorize_header_batch(raw_objs: Sequence[dict]) -> np.ndarray:
    n = len(raw_objs)
    out = np.zeros((n, HEADER_DIM), dtype=np.float32)
    for i, raw_obj in enumerate(raw_objs):
        if not raw_obj:
            continue
        coff = raw_obj["coff"]
        optional = raw_obj["optional"]
        dos = raw_obj["dos"]
        row = np.hstack(
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
        out[i] = row
    return out


def vectorize_imports_batch(raw_objs: Sequence[dict]) -> np.ndarray:
    n = len(raw_objs)
    out = np.zeros((n, IMPORTS_DIM), dtype=np.float32)

    lib_docs: List[List[str]] = []
    import_docs: List[List[str]] = []
    lengths = np.zeros((n, 2), dtype=np.float32)
    nonempty_idx: List[int] = []

    for i, raw_obj in enumerate(raw_objs):
        if not raw_obj:
            lib_docs.append([])
            import_docs.append([])
            continue
        libraries = list({lib.lower() for lib in raw_obj.keys()})
        imports = [lib.lower() + ":" + entry for lib, items in raw_obj.items() for entry in items]
        lib_docs.append(libraries)
        import_docs.append(imports)
        lengths[i, 0] = len(imports)
        lengths[i, 1] = len(libraries)
        nonempty_idx.append(i)

    if nonempty_idx:
        libs_hashed = _LIB_HASHER.transform(lib_docs).toarray().astype(np.float32)
        imports_hashed = _IMPORT_HASHER.transform(import_docs).toarray().astype(np.float32)
        out[:, 0:2] = lengths
        out[:, 2:258] = libs_hashed
        out[:, 258:1282] = imports_hashed
    return out
