from __future__ import annotations

"""
MaxCompute SavedDataset Storage
================================
Allows Feast to persist SavedDatasets (historical feature retrieval results)
back into a MaxCompute table for later reuse in model training.
"""

from typing import Optional

from feast.saved_dataset import SavedDatasetStorage
from feast.protos.feast.core.SavedDataset_pb2 import (
    SavedDatasetStorage as SavedDatasetStorageProto,
)


class MaxComputeSavedDatasetOptions:
    """Options that identify the destination MaxCompute table."""

    def __init__(self, table: str, project: Optional[str] = None):
        self.table = table
        self.project = project  # falls back to offline_store.project if None

    def __repr__(self) -> str:
        return f"MaxComputeSavedDatasetOptions(table={self.table!r})"


class MaxComputeSavedDatasetStorage(SavedDatasetStorage):
    """
    SavedDatasetStorage backed by a MaxCompute table.

    Usage:
        storage = MaxComputeSavedDatasetStorage(table="training_dataset_v1")
        store.create_saved_dataset(
            from_=job,
            name="training_v1",
            storage=storage,
        )
    """

    _proto_attr_name = "custom_storage"

    def __init__(self, table: str, project: Optional[str] = None):
        self.maxcompute_options = MaxComputeSavedDatasetOptions(
            table=table, project=project
        )

    def to_proto(self) -> SavedDatasetStorageProto:
        import json

        return SavedDatasetStorageProto(
            custom_storage=SavedDatasetStorageProto.CustomStorageOptions(
                configuration=json.dumps(
                    {
                        "table": self.maxcompute_options.table,
                        "project": self.maxcompute_options.project,
                    }
                ).encode("utf-8")
            )
        )

    @classmethod
    def from_proto(
        cls, storage_proto: SavedDatasetStorageProto
    ) -> "MaxComputeSavedDatasetStorage":
        import json

        cfg = json.loads(
            storage_proto.custom_storage.configuration.decode("utf-8")
        )
        return cls(table=cfg["table"], project=cfg.get("project"))

    def to_data_source(self):
        from feast_maxcompute.maxcompute_source import MaxComputeSource

        return MaxComputeSource(
            name=self.maxcompute_options.table,
            table=self.maxcompute_options.table,
            mc_project=self.maxcompute_options.project or "",
        )