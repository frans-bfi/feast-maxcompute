from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pyarrow as pa
from feast.data_source import DataSource
from feast.protos.feast.core.DataSource_pb2 import DataSource as DataSourceProto
from feast.repo_config import RepoConfig
from feast.type_map import feast_value_type_to_pa
from feast.value_type import ValueType
from google.protobuf.json_format import MessageToJson


# Mapping from MaxCompute/ODPS types → Feast ValueType
MAXCOMPUTE_TO_FEAST_TYPE: Dict[str, ValueType] = {
    "BIGINT": ValueType.INT64,
    "INT": ValueType.INT32,
    "SMALLINT": ValueType.INT32,
    "TINYINT": ValueType.INT32,
    "FLOAT": ValueType.FLOAT,
    "DOUBLE": ValueType.DOUBLE,
    "DECIMAL": ValueType.DOUBLE,
    "VARCHAR": ValueType.STRING,
    "CHAR": ValueType.STRING,
    "STRING": ValueType.STRING,
    "BOOLEAN": ValueType.BOOL,
    "DATETIME": ValueType.UNIX_TIMESTAMP,
    "TIMESTAMP": ValueType.UNIX_TIMESTAMP,
    "DATE": ValueType.UNIX_TIMESTAMP,
    "BINARY": ValueType.BYTES,
    "ARRAY<BIGINT>": ValueType.INT64_LIST,
    "ARRAY<INT>": ValueType.INT32_LIST,
    "ARRAY<FLOAT>": ValueType.FLOAT_LIST,
    "ARRAY<DOUBLE>": ValueType.DOUBLE_LIST,
    "ARRAY<STRING>": ValueType.STRING_LIST,
    "ARRAY<BOOLEAN>": ValueType.BOOL_LIST,
    "ARRAY<BINARY>": ValueType.BYTES_LIST,
}

# Mapping from MaxCompute types → PyArrow types
MAXCOMPUTE_TO_ARROW_TYPE: Dict[str, pa.DataType] = {
    "BIGINT": pa.int64(),
    "INT": pa.int32(),
    "SMALLINT": pa.int16(),
    "TINYINT": pa.int8(),
    "FLOAT": pa.float32(),
    "DOUBLE": pa.float64(),
    "DECIMAL": pa.float64(),
    "VARCHAR": pa.string(),
    "CHAR": pa.string(),
    "STRING": pa.string(),
    "BOOLEAN": pa.bool_(),
    "DATETIME": pa.timestamp("us"),
    "TIMESTAMP": pa.timestamp("us"),
    "DATE": pa.date32(),
    "BINARY": pa.binary(),
}


class MaxComputeSource(DataSource):
    """
    Custom Feast DataSource backed by Alibaba MaxCompute (ODPS).

    Example usage in a feature repo:
        source = MaxComputeSource(
            name="driver_stats_source",
            table="driver_stats",
            mc_project="my_mc_project",
            timestamp_field="event_timestamp",
            created_timestamp_column="created",
        )
    """
    DATA_SOURCE_CLASS_TYPE = "feast_maxcompute.maxcompute_source.MaxComputeSource"
    
    def __init__(
        self,
        *,
        name: str,
        table: str,
        mc_project: str,
        timestamp_field: str = "",
        created_timestamp_column: str = "",
        field_mapping: Optional[Dict[str, str]] = None,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        owner: str = "",
        query: Optional[str] = None,
        partitioned: bool = False,
    ):
        """
        Args:
            name:                     Feast source name.
            table:                    MaxCompute table name (without project prefix).
            mc_project:               MaxCompute project name that owns the table.
            timestamp_field:          Column used as the event timestamp.
            created_timestamp_column: Column used as the row-creation timestamp.
            field_mapping:            Optional rename map {mc_col: feast_col}.
            description:              Human-readable description.
            tags:                     Arbitrary key-value metadata.
            owner:                    Owner email / identifier.
            query:                    Optional raw SQL override (uses table if not set).
            partitioned:             Whether the source is partitioned.
        """
        self._table = table
        self._mc_project = mc_project
        self._query = query
        self._partitioned = partitioned

        super().__init__(
            name=name,
            timestamp_field=timestamp_field,
            created_timestamp_column=created_timestamp_column,
            field_mapping=field_mapping or {},
            description=description,
            tags=tags or {},
            owner=owner,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def table(self) -> str:
        return self._table

    @property
    def mc_project(self) -> str:
        return self._mc_project

    @property
    def query(self) -> Optional[str]:
        return self._query

    # ------------------------------------------------------------------
    # Required DataSource interface
    # ------------------------------------------------------------------

    def get_table_query_string(self
    ) -> str:
        """Return a SQL-compatible reference to the underlying data."""
        if self._query:
            return f"({self._query})"
        
        table_ref = f"`{self._mc_project}`.`{self._table}`"
        if self._partitioned:
            from datetime import datetime, timedelta
            yestedray_str = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
            # yestedray_str = '20260525'
            filters = f"`dt` = '{yestedray_str}'"
            return f"(SELECT * FROM {table_ref} WHERE {filters})"
        
        return table_ref

    @staticmethod
    def source_datatype_to_feast_value_type() -> Callable[[str], ValueType]:
        """Return a function that maps MaxCompute column types to Feast ValueType."""

        def mapper(mc_type: str) -> ValueType:
            normalised = mc_type.upper().strip()
            # Handle parameterised types like DECIMAL(10,2) or VARCHAR(255)
            base = normalised.split("(")[0].strip()
            return MAXCOMPUTE_TO_FEAST_TYPE.get(base, ValueType.UNKNOWN)

        return mapper

    def get_table_column_names_and_types(
        self, config: RepoConfig
    ) -> Iterable[Tuple[str, str]]:
        """
        Introspect the MaxCompute table schema and return (column_name, mc_type) pairs.
        Requires a live ODPS connection built from config.
        """
        from feast_maxcompute.maxcompute_offline_store import _get_odps_client

        odps = _get_odps_client(config.offline_store)
        tbl = odps.get_table(self._table, project=self._mc_project)
        schema = tbl.table_schema
        columns = list(schema.columns)
        if schema.partitions:
            columns += list(schema.partitions)
        return [(col.name, col.type.name.upper()) for col in columns]

    # ------------------------------------------------------------------
    # Serialisation helpers (proto round-trip)
    # ------------------------------------------------------------------

    def to_proto(self) -> DataSourceProto:
        """Serialise to Feast's DataSource protobuf (stored in the registry)."""
        return DataSourceProto(
            name=self.name,
            type=DataSourceProto.CUSTOM_SOURCE,
            data_source_class_type=self.DATA_SOURCE_CLASS_TYPE,
            field_mapping=self.field_mapping,
            timestamp_field=self.timestamp_field,
            created_timestamp_column=self.created_timestamp_column,
            description=self.description,
            tags=self.tags,
            owner=self.owner,
            custom_options=DataSourceProto.CustomSourceOptions(
                configuration=self._serialise_custom_config()
            ),
        )

    @classmethod
    def from_proto(cls, data_source: DataSourceProto) -> "MaxComputeSource":
        """Deserialise from Feast's DataSource protobuf."""
        import json

        custom_cfg: Dict[str, Any] = json.loads(
            data_source.custom_options.configuration.decode("utf-8")
        )
        # Restore partition configs
        partitioned = custom_cfg.get("partitioned", False)
         
        return cls(
            name=data_source.name,
            table=custom_cfg.get("table"),
            mc_project=custom_cfg["mc_project"],
            timestamp_field=data_source.timestamp_field,
            created_timestamp_column=data_source.created_timestamp_column,
            field_mapping=dict(data_source.field_mapping),
            description=data_source.description,
            tags=dict(data_source.tags),
            owner=data_source.owner,
            query=custom_cfg.get("query"),
            partitioned=partitioned,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _serialise_custom_config(self) -> bytes:
        import json

        cfg = {"table": self._table, "mc_project": self._mc_project}
        cfg["partitioned"] = self._partitioned
        if self._query:
            cfg["query"] = self._query
        
        return json.dumps(cfg).encode("utf-8")

    def validate(self, config: RepoConfig) -> None:
        """Validate that the referenced table exists."""
        self.get_table_column_names_and_types(config)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MaxComputeSource):
            return False
        return (
            self.name == other.name
            and self._table == other._table
            and self._mc_project == other._mc_project
            and self._query == other._query
            and self.timestamp_field == other.timestamp_field
        )

    def __hash__(self) -> int:
        return hash((self.name, self._table, self._mc_project))

    def __repr__(self) -> str:
        return (
            f"MaxComputeSource(name={self.name!r}, "
            f"mc_project={self._mc_project!r}, table={self._table!r})"
        )
