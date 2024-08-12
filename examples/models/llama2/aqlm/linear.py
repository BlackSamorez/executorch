""" AQLM 2x8 Linear"""
import torch
import torch.nn as nn

class Aqlm2x8Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_group_size: int,
        out_group_size: int,
        num_codebooks: int,
        nbits_per_codebook: int,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        assert self.in_features % in_group_size == 0
        assert self.out_features % out_group_size == 0
        num_out_groups = out_features // out_group_size
        num_in_groups = in_features // in_group_size
        self.out_group_size, self.in_group_size = out_group_size, in_group_size
        self.num_codebooks = num_codebooks
        self.nbits_per_codebook = nbits_per_codebook
        self.codebook_size = 2**nbits_per_codebook

        # CODES & CODEBOOKS
        self.codebooks = nn.Parameter(
            torch.empty((num_codebooks, self.codebook_size, out_group_size, in_group_size), **factory_kwargs) * 2 - 1,
            requires_grad=False,
        )  # [num_codebooks, codebook_size, out_group_size, in_group_size]
        self.codes = nn.Parameter(
            torch.empty(
                (num_in_groups, num_out_groups, num_codebooks),
                device=device,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )  #  [num_in_groups, num_out_groups, num_codebooks]

        # SCALES
        self.scales = nn.Parameter(
            torch.empty((num_out_groups, 1, 1, 1), **factory_kwargs), requires_grad=False
        )  #  [num_out_groups, 1, 1, 1]

        # BIAS
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.ops.aqlm.code2x8_lut_matmat(
            input, self.codes, self.codebooks, self.scales, bias=self.bias
        )
