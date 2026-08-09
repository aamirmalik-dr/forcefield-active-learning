"""falearn: query-by-committee active learning for TiZrNb interatomic potentials.

A from-scratch deep ensemble of compact ACSF potentials, a label-budget
benchmark of committee-disagreement acquisition against random selection, and
an uncertainty-calibration analysis. Training labels are surrogate model
labels from a pretrained universal potential, never DFT; real DFT enters only
as Materials Project validation anchors.
"""

from falearn.acquisition import select_random, select_top_disagreement
from falearn.data import Frame, load_frames, pool_test_split, save_frames
from falearn.descriptors import AcsfParams, build_graph, frame_graph
from falearn.ensemble import committee_predict, evaluate_mean, train_committee
from falearn.loop import EvalSet, LoopSettings, run_loop
from falearn.model import MemberPotential, load_members, save_members
from falearn.pool import LabeledPool
from falearn.training import TrainSettings

__version__ = "0.1.0"

__all__ = [
    "AcsfParams",
    "EvalSet",
    "Frame",
    "LabeledPool",
    "LoopSettings",
    "MemberPotential",
    "TrainSettings",
    "build_graph",
    "committee_predict",
    "evaluate_mean",
    "frame_graph",
    "load_frames",
    "load_members",
    "pool_test_split",
    "run_loop",
    "save_frames",
    "save_members",
    "select_random",
    "select_top_disagreement",
    "train_committee",
]
