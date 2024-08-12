import os
from typing import Optional
import logging
from pathlib import Path

import torch
from torch.library import impl

try:
    op = torch.ops.aqlm.code2x8_lut_matmat.default
    assert op is not None
except:
    libs = list(Path(__file__).parent.resolve().glob("libaqlm_aot_lib.*"))
    assert len(libs) == 1, f"Expected 1 library but got {len(libs)}"
    logging.warn(f"Loading aqlm library: {libs[0]}")
    torch.ops.load_library(libs[0])
    op = torch.ops.aqlm.code2x8_lut_matmat.default
    assert op is not None

aqlm_lib = torch.library.Library("aqlm", "IMPL")

@impl(aqlm_lib, "code2x8_lut_matmat", "Meta")
def code2x8_lut_matmat_meta(input, codes, codebooks, scales, bias=None):
    return torch.empty(
        input.shape[:-1] + (codes.shape[1],), device=input.device, dtype=input.dtype
    )
