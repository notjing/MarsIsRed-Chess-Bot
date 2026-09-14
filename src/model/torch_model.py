"""V26 dual-head net in PyTorch.

Matches the live Keras graph (not aimodel.py): 8 resblocks, WDL softmax,
policy 1x1-conv-2 -> BN -> NHWC flatten -> Dense 4672 logits.

Internal layout is NCHW. Flatten is permuted back to NHWC so Keras Dense
weights stay valid. Search still sees NHWC ONNX via NHWCChessNet.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_EPS = 1e-3
BN_MOMENTUM = 0.01
L2 = 5e-5
FILTERS = 256
N_BLOCKS = 8
POLICY_PLANES = 73
BOARD_PLANES = 25
EXTRA_DIM = 19
WDL_DIM = 3
POLICY_DIM = 8 * 8 * POLICY_PLANES


def _flatten_nhwc(x: torch.Tensor) -> torch.Tensor:
    """Keras Flatten on channels-last: (N, C, H, W) -> (N, H*W*C)."""
    return x.permute(0, 2, 3, 1).contiguous().flatten(1)


class ResBlock(nn.Module):
    """Conv-ReLU-BN, Conv-BN, skip, ReLU. Matches Keras res_block."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(FILTERS, FILTERS, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(FILTERS, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(FILTERS, FILTERS, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(FILTERS, eps=BN_EPS, momentum=BN_MOMENTUM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return F.relu(x + shortcut)


class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(BOARD_PLANES, FILTERS, 3, padding=1, bias=True)
        self.blocks = nn.ModuleList([ResBlock() for _ in range(N_BLOCKS)])

        self.value_conv = nn.Conv2d(FILTERS, 32, 1, bias=True)
        self.value_fc128 = nn.Linear(8 * 8 * 32, 128)
        self.value_fc64 = nn.Linear(128 + EXTRA_DIM, 64)
        self.value_fc32 = nn.Linear(64, 32)
        self.prob_dist = nn.Linear(32, WDL_DIM)

        self.policy_conv = nn.Conv2d(FILTERS, 2, 1, bias=True)
        self.policy_bn = nn.BatchNorm2d(2, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.move_dist = nn.Linear(8 * 8 * 2, POLICY_DIM)

        # Keras kernel L2 on these layers only (stem + output heads excluded).
        self._l2_kernels = (
            [blk.conv1.weight for blk in self.blocks]
            + [blk.conv2.weight for blk in self.blocks]
            + [
                self.value_conv.weight,
                self.value_fc128.weight,
                self.value_fc64.weight,
                self.value_fc32.weight,
                self.policy_conv.weight,
            ]
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            freeze_batchnorm(self)
        return self

    def forward(self, board: torch.Tensor, extra: torch.Tensor):
        x = F.relu(self.stem(board))
        for block in self.blocks:
            x = block(x)

        v = F.relu(self.value_conv(x))
        v = _flatten_nhwc(v)
        v = F.relu(self.value_fc128(v))
        v = torch.cat([v, extra], dim=1)
        v = F.relu(self.value_fc64(v))
        v = self.value_fc32(v)
        v = v * torch.sigmoid(v)
        wdl = torch.softmax(self.prob_dist(v), dim=-1)

        p = F.relu(self.policy_conv(x))
        p = self.policy_bn(p)
        p = _flatten_nhwc(p)
        logits = self.move_dist(p)
        return wdl, logits

    def l2_loss(self) -> torch.Tensor:
        loss = self.stem.weight.new_zeros(())
        for w in self._l2_kernels:
            loss = loss + w.square().sum()
        return L2 * loss


class NHWCChessNet(nn.Module):
    """ONNX-facing wrapper: board_input is (N, 8, 8, 25) like Keras/ORT."""

    def __init__(self, net: ChessNet):
        super().__init__()
        self.net = net

    def forward(self, board_input: torch.Tensor, extra_input: torch.Tensor):
        board = board_input.permute(0, 3, 1, 2).contiguous()
        return self.net(board, extra_input)


def freeze_batchnorm(model: nn.Module) -> None:
    """Keras trainable=False on BN: frozen gamma/beta, frozen running stats."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            if m.weight is not None:
                m.weight.requires_grad_(False)
            if m.bias is not None:
                m.bias.requires_grad_(False)


def export_onnx(model: ChessNet, path: str, opset: int = 13) -> None:
    wrapper = NHWCChessNet(model).eval()
    dummy_board = torch.zeros(1, 8, 8, BOARD_PLANES, dtype=torch.float32)
    dummy_extra = torch.zeros(1, EXTRA_DIM, dtype=torch.float32)
    kwargs = {
        "input_names": ["board_input", "extra_input"],
        "output_names": ["prob_dist", "move_dist"],
        "dynamic_axes": {
            "board_input": {0: "batch"},
            "extra_input": {0: "batch"},
            "prob_dist": {0: "batch"},
            "move_dist": {0: "batch"},
        },
        "opset_version": opset,
    }
    sig = inspect.signature(torch.onnx.export)
    if "dynamo" in sig.parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(wrapper, (dummy_board, dummy_extra), path, **kwargs)
