"""
pt_multitask.py

Multi-task PointTransformer.  Subclasses the single-task PointTransformer
from pt.py and replaces the single classifier head with two: one for
ToolType (cls_tool) and one for ModuleType (cls_module).  Both heads
operate on the same final point-feature tensor x1 produced by the
shared encoder/decoder.

Pretrained loading:
    The Phase 1 PointTransformer checkpoint has keys like 'enc1_td.*',
    'dec1_block.*', 'cls.0.weight', etc.  When loaded into this
    multi-task model:
      - encoder/decoder keys match exactly and load
      - 'cls.*' keys have no matching destination -> skipped
      - 'cls_tool.*' and 'cls_module.*' are not in the checkpoint
        -> random initialization (as we want)

Forward signature:
    Returns (tool_log_probs, module_log_probs) where each has shape
    [B, num_classes_<head>, N].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pt import PointTransformer


# Width of the final point-feature tensor x1, taken from PartSeg26 config
# (planes[0] = 32 in pt.py).  If pt.py is reconfigured this needs updating.
SHARED_FEATURE_DIM = 32


class PointTransformerMultiTask(PointTransformer):
    """Two-head PointTransformer for joint ToolType + ModuleType segmentation."""

    def __init__(self, num_classes_tool, num_classes_module,
                 num_points=4096, k=None):
        # super() builds enc*, dec*, and self.cls with num_classes=num_classes_tool.
        super().__init__(num_classes=num_classes_tool, num_points=num_points, k=k)

        # Rename the inherited head to make the two-head structure explicit.
        # del self.cls removes the submodule registration so it doesn't appear
        # in state_dict.
        self.cls_tool = self.cls
        del self.cls

        # Module head -- same architecture as cls_tool, different output size.
        d = SHARED_FEATURE_DIM
        self.cls_module = nn.Sequential(
            nn.Conv1d(d, d, 1, bias=True),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True),
            nn.Conv1d(d, num_classes_module, 1, bias=True),
        )

    def forward(self, pos, x):
        """Returns (tool_log_probs, module_log_probs).
        Each is [B, num_classes_*, N]."""
        # Encoder -- identical to PointTransformer.forward
        p1, x1 = self.enc1_td(pos, x)
        x1 = self.enc1_block(p1, x1)
        p2, x2 = self.enc2_td(p1, x1)
        x2 = self.enc2_block(p2, x2)
        p3, x3 = self.enc3_td(p2, x2)
        x3 = self.enc3_block(p3, x3)
        p4, x4 = self.enc4_td(p3, x3)
        x4 = self.enc4_block(p4, x4)
        p5, x5 = self.enc5_td(p4, x4)
        x5 = self.enc5_block(p5, x5)

        # Decoder -- identical
        x5 = self.dec5_head(p5, x5)
        x5 = self.dec5_block(p5, x5)
        x4 = self.dec4_up(p4, x4, p5, x5)
        x4 = self.dec4_block(p4, x4)
        x3 = self.dec3_up(p3, x3, p4, x4)
        x3 = self.dec3_block(p3, x3)
        x2 = self.dec2_up(p2, x2, p3, x3)
        x2 = self.dec2_block(p2, x2)
        x1 = self.dec1_up(p1, x1, p2, x2)
        x1 = self.dec1_block(p1, x1)

        # Two heads applied to the same x1
        tool_logits   = self.cls_tool(x1)     # [B, num_classes_tool, N]
        module_logits = self.cls_module(x1)   # [B, num_classes_module, N]

        return (F.log_softmax(tool_logits,   dim=1),
                F.log_softmax(module_logits, dim=1))
