"""Array providers. Model code stays in ``rho_cca.extract`` and is optional."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from rho_cca.records import Endpoint, EndpointMismatch, FeatureTable, ModelPair, PairedDataset, RhoCCAError, Split


class ArrayProvider:
    """Look up a precomputed table by feature identity, then by sample id."""

    def __init__(self, tables: Sequence[FeatureTable]):
        self.tables = {table.endpoint.identity_key(): table for table in tables}

    def extract(
        self,
        endpoint: Endpoint,
        sample_ids: Sequence[str],
        group_ids: Sequence[str] | None = None,
    ) -> FeatureTable:
        table = self.tables.get(endpoint.identity_key())
        if table is None:
            raise EndpointMismatch(f"no array stored for {endpoint.model}@{endpoint.revision} layer {endpoint.layer}")
        taken = table.take(table.index(sample_ids))
        if group_ids is not None and tuple(str(item) for item in group_ids) != taken.group_ids:
            raise RhoCCAError("provider group ids do not match the stored table")
        if taken.endpoint.width != endpoint.width:
            raise EndpointMismatch("stored width does not match the requested endpoint")
        return taken


class MapExtractor:
    """Call a per-basis function. Suitable for tests and for a thin model wrapper."""

    def __init__(self, functions: dict[str, Callable[[Sequence, Endpoint], np.ndarray]]):
        self.functions = dict(functions)

    def extract(
        self,
        endpoint: Endpoint,
        sample_ids: Sequence[str],
        group_ids: Sequence[str],
        inputs: Sequence,
    ) -> FeatureTable:
        if len(inputs) != len(sample_ids) or len(group_ids) != len(sample_ids):
            raise RhoCCAError("inputs, sample ids, and group ids must align")
        function = self.functions.get(endpoint.identity_key())
        if function is None:
            raise EndpointMismatch(f"no extractor registered for {endpoint.identity_key()}")
        values = np.asarray(function(inputs, endpoint), dtype=np.float64)
        return FeatureTable(endpoint, tuple(str(item) for item in sample_ids), tuple(str(item) for item in group_ids), values)


def collect_pair(
    model_pair: ModelPair,
    provider: ArrayProvider | MapExtractor,
    sample_ids: Sequence[str],
    group_ids: Sequence[str],
    *,
    source_inputs: Sequence | None = None,
    partner_inputs: Sequence | None = None,
    split: Split | None = None,
) -> PairedDataset:
    """Build a paired dataset by asking the provider for both endpoints on the model pair."""
    if isinstance(provider, MapExtractor):
        if source_inputs is None or partner_inputs is None:
            raise RhoCCAError("a map extractor needs inputs for both endpoints")
        source = provider.extract(model_pair.source, sample_ids, group_ids, source_inputs)
        partner = provider.extract(model_pair.partner, sample_ids, group_ids, partner_inputs)
    else:
        source = provider.extract(model_pair.source, sample_ids, group_ids)
        partner = provider.extract(model_pair.partner, sample_ids, group_ids)
    return PairedDataset(source, partner, split)
