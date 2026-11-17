"""CPU self-check for the shared-subspace helpers used by the sentiment runs."""

from prh_replication.multifamily_shared_subspace import _self_check


def test_shared_subspace_self_check():
    _self_check()
