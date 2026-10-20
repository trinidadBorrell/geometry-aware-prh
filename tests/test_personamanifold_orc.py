"""Our all-edges ORC on PersonaManifold's graph equals their per-edge `edge_curvature`."""

import sys

import networkx as nx
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="GraphRicciCurvature needs the fork start method (Linux)"
)


def test_matches_their_edge_curvature():
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
    from personamanifold import PersonaManifold
    from personamanifold.curvature import edge_curvature

    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 3))
    pm = PersonaManifold(n_neighbors=5, variance=1.0, tangent_dim=3, ridge=1e-4).fit(x)
    g = nx.from_scipy_sparse_array(pm.graph_, edge_attribute="weight")
    orc = OllivierRicci(g, alpha=0.0, method="OTD", exp_power=0, proc=1, verbose="ERROR")
    orc.compute_ricci_curvature()
    for u, v in list(orc.G.edges())[:15]:
        assert orc.G[u][v]["ricciCurvature"] == pytest.approx(edge_curvature(pm, u, v), abs=1e-6)
