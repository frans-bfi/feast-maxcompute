from __future__ import annotations

"""
MaxCompute (ODPS) Offline Store for Feast
==========================================
Drop-in offline store that connects Feast to Alibaba Cloud MaxCompute / ODPS.
"""

import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
from pydantic import StrictStr
from typing_extensions import Literal

from feast.feature_logging import LoggingConfig, LoggingSource
from feast.feature_view import FeatureView
from feast.infra.offline_stores.offline_store import (
    OfflineStore,
    RetrievalJob,
    RetrievalMetadata,
)
from feast.infra.registry.base_registry import BaseRegistry
from feast.repo_config import FeastConfigBaseModel, RepoConfig
from feast.saved_dataset import SavedDatasetStorage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class MaxComputeOfflineStoreConfig(FeastConfigBaseModel):
    """
    All fields map 1-to-1 to pyodps.ODPS() constructor arguments.

    feature_store.yaml example:
        offline_store:
            type: feast_maxcompute.maxcompute_offline_store.MaxComputeOfflineStore
            access_id: "LTAI5tXXXXXX"
            secret_access_key: "XXXXXXXXXXXXXXXXXXXXXXXX"
            project: "my_mc_project"
            endpoint: "http://service.cn-hangzhou.maxcompute.aliyun.com/api"
            tunnel_endpoint: "http://dt.cn-hangzhou.maxcompute.aliyun.com"   # optional
            temp_table_prefix: "feast_tmp_"                                  # optional
            result_limit: 1000000                                            # optional
    """

    type: Literal[
        "feast_maxcompute.maxcompute_offline_store.MaxComputeOfflineStore"
    ] = "feast_maxcompute.maxcompute_offline_store.MaxComputeOfflineStore"

    # Required MaxCompute credentials
    access_id: StrictStr
    secret_access_key: StrictStr
    project: StrictStr
    endpoint: StrictStr  # e.g. "http://service.cn-hangzhou.maxcompute.aliyun.com/api"

    # Optional
    tunnel_endpoint: Optional[StrictStr] = None
    temp_table_prefix: StrictStr = "feast_tmp_"
    result_limit: int = 1_000_000


# ---------------------------------------------------------------------------
# ODPS client helper
# ---------------------------------------------------------------------------

def _get_odps_client(store_config: MaxComputeOfflineStoreConfig):
    """Return an authenticated pyodps ODPS client."""
    try:
        from odps import ODPS
    except ImportError as e:
        raise ImportError(
            "pyodps is required for MaxComputeOfflineStore. "
            "Install it with: pip install pyodps"
        ) from e

    kwargs: Dict[str, Any] = {
        "access_id": store_config.access_id,
        "secret_access_key": store_config.secret_access_key,
        "project": store_config.project,
        "endpoint": store_config.endpoint,
    }
    if store_config.tunnel_endpoint:
        kwargs["tunnel_endpoint"] = store_config.tunnel_endpoint

    return ODPS(**kwargs)


# ---------------------------------------------------------------------------
# RetrievalJob
# ---------------------------------------------------------------------------

class MaxComputeRetrievalJob(RetrievalJob):
    """
    Wraps a MaxCompute SQL query and materialises results on demand.
    Execution is deferred — the query only runs when you call
    .to_df() / .to_arrow() / .persist().
    """

    def __init__(
        self,
        query: str,
        store_config: MaxComputeOfflineStoreConfig,
        full_feature_names: bool,
        metadata: Optional[RetrievalMetadata] = None,
        on_demand_feature_views: Optional[List[Any]] = None,
    ):
        self._query = query
        self._store_config = store_config
        self._full_feature_names = full_feature_names
        self._metadata = metadata
        self._on_demand_feature_views = on_demand_feature_views or []

    @property
    def full_feature_names(self) -> bool:
        return self._full_feature_names

    @property
    def on_demand_feature_views(self) -> List[Any]:
        return self._on_demand_feature_views

    @property
    def metadata(self) -> Optional[RetrievalMetadata]:
        return self._metadata

    def _to_df_internal(self, timeout: Optional[int] = None) -> pd.DataFrame:
        odps = _get_odps_client(self._store_config)
        logger.debug("Executing MaxCompute SQL:\n%s", self._query)
        with odps.execute_sql(self._query).open_reader(tunnel=True) as reader:
            df = reader.to_pandas()
        return _cast_timestamp_columns(df)

    def _to_arrow_internal(self, timeout: Optional[int] = None) -> pa.Table:
        return pa.Table.from_pandas(self._to_df_internal(timeout=timeout))

    def persist(
        self,
        storage: SavedDatasetStorage,
        allow_overwrite: bool = False,
        timeout: Optional[int] = None,
    ) -> None:
        """Persist the result into a MaxCompute table for SavedDatasets."""
        from feast_maxcompute.maxcompute_saved_dataset_storage import (
            MaxComputeSavedDatasetStorage,
        )

        assert isinstance(storage, MaxComputeSavedDatasetStorage), (
            "storage must be MaxComputeSavedDatasetStorage"
        )

        odps = _get_odps_client(self._store_config)
        dest_table = storage.maxcompute_options.table

        if allow_overwrite:
            odps.execute_sql(
                f"INSERT OVERWRITE TABLE `{dest_table}` {self._query}"
            )
        else:
            odps.execute_sql(
                f"CREATE TABLE IF NOT EXISTS `{dest_table}` AS {self._query}"
            )

    def __repr__(self) -> str:
        return f"MaxComputeRetrievalJob(query={self._query[:80]!r}...)"


# ---------------------------------------------------------------------------
# OfflineStore
# ---------------------------------------------------------------------------

class MaxComputeOfflineStore(OfflineStore):
    """
    Feast OfflineStore implementation for Alibaba MaxCompute (ODPS).

    Register in feature_store.yaml:
        offline_store:
            type: feast_maxcompute.maxcompute_offline_store.MaxComputeOfflineStore
    """

    @staticmethod
    def get_historical_features(
        config: RepoConfig,
        feature_views: List[FeatureView],
        feature_refs: List[str],
        entity_df: Union[pd.DataFrame, str, None],
        registry: BaseRegistry,
        project: str,
        full_feature_names: bool = False,
        **kwargs,
    ) -> MaxComputeRetrievalJob:
        store_cfg: MaxComputeOfflineStoreConfig = config.offline_store

        # Optional date range — accepted as kwargs so the method signature
        # stays compatible with Feast's OfflineStore base class.
        start_date: Optional[datetime] = kwargs.get("start_date")
        end_date: Optional[datetime] = kwargs.get("end_date")

        # ------------------------------------------------------------------
        # Route A: no entity_df provided — use start_date/end_date to do a
        # full range scan across all entities (pull_all logic per feature view).
        # Useful when you want all data in a window without a fixed entity list.
        # ------------------------------------------------------------------
        if entity_df is None:
            if not start_date or not end_date:
                raise ValueError(
                    "When entity_df is None, both start_date and end_date "
                    "must be provided as keyword arguments.\n"
                    "Example:\n"
                    "    store.get_historical_features(\n"
                    "        entity_df=None,\n"
                    "        features=[...],\n"
                    "        start_date=datetime(2025, 7, 1),\n"
                    "        end_date=datetime(2025, 7, 2),\n"
                    "    )"
                )

            # Ensure dates are timezone-aware
            start_date = _ensure_utc(start_date)
            end_date   = _ensure_utc(end_date)

            query = _build_range_scan_query(
                config=config,
                feature_views=feature_views,
                feature_refs=feature_refs,
                start_date=start_date,
                end_date=end_date,
                full_feature_names=full_feature_names,
            )

            retrieval_metadata = _get_retrieval_metadata(
                registry=registry,
                feature_views=feature_views,
                feature_refs=feature_refs,
                project=project,
                full_feature_names=full_feature_names,
            )

            return MaxComputeRetrievalJob(
                query=query,
                store_config=store_cfg,
                full_feature_names=full_feature_names,
                metadata=retrieval_metadata,
            )

        # ------------------------------------------------------------------
        # Route B: entity_df provided — standard point-in-time join.
        # start_date/end_date are ignored here; the range is defined by the
        # event_timestamp values inside entity_df.
        # ------------------------------------------------------------------
        odps = _get_odps_client(store_cfg)
        entity_table_name = _upload_entity_df(odps, store_cfg, entity_df)

        query = _build_point_in_time_query(
            config=config,
            feature_views=feature_views,
            feature_refs=feature_refs,
            entity_table_name=entity_table_name,
            full_feature_names=full_feature_names,
        )
        retrieval_metadata = _get_retrieval_metadata(
            registry=registry,
            feature_views=feature_views,
            feature_refs=feature_refs,
            project=project,
            full_feature_names=full_feature_names,
        )

        return MaxComputeRetrievalJob(
            query=query,
            store_config=store_cfg,
            full_feature_names=full_feature_names,
            metadata=retrieval_metadata,
        )

    @staticmethod
    def pull_latest_from_table_or_query(
        config: RepoConfig,
        data_source: Any,
        join_key_columns: List[str],
        feature_name_columns: List[str],
        timestamp_field: str,
        created_timestamp_column: Optional[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs,
    ) -> MaxComputeRetrievalJob:
        store_cfg: MaxComputeOfflineStoreConfig = config.offline_store
        from feast_maxcompute.maxcompute_source import MaxComputeSource

        assert isinstance(data_source, MaxComputeSource), (
            "pull_latest_from_table_or_query requires a MaxComputeSource"
        )

        # Use partition-aware reference — injects partition WHERE clauses
        # so MaxCompute can prune partitions before scanning rows.
        table_ref = data_source.get_table_query_string()

        partition_by = ", ".join(join_key_columns)
        all_columns = join_key_columns + feature_name_columns + [timestamp_field]
        if created_timestamp_column:
            all_columns.append(created_timestamp_column)

        order_clause = (
            f"ORDER BY `{timestamp_field}` DESC, `{created_timestamp_column}` DESC"
            if created_timestamp_column
            else f"ORDER BY `{timestamp_field}` DESC"
        )

        start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

        query = f"""
SELECT {_col_list(all_columns)}
FROM (
    SELECT
        {_col_list(all_columns)},
        ROW_NUMBER() OVER (
            PARTITION BY {_col_list(join_key_columns)}
            {order_clause}
        ) AS _feast_row
    FROM {table_ref}
    WHERE
        `{timestamp_field}` >= TIMESTAMP '{start_str}'
        AND `{timestamp_field}` < TIMESTAMP '{end_str}'
) t
WHERE _feast_row = 1
        """.strip()

        return MaxComputeRetrievalJob(
            query=query,
            store_config=store_cfg,
            full_feature_names=False,
        )

    @staticmethod
    def pull_all_from_table_or_query(
        config: RepoConfig,
        data_source: Any,
        join_key_columns: List[str],
        feature_name_columns: List[str],
        timestamp_field: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs,
    ) -> MaxComputeRetrievalJob:
        store_cfg: MaxComputeOfflineStoreConfig = config.offline_store
        from feast_maxcompute.maxcompute_source import MaxComputeSource

        assert isinstance(data_source, MaxComputeSource), (
            "pull_all_from_table_or_query requires a MaxComputeSource"
        )

        # Partition-aware table reference for efficient partition pruning
        table_ref = data_source.get_table_query_string()

        all_columns = join_key_columns + feature_name_columns + [timestamp_field]
        start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

        query = f"""
SELECT {_col_list(all_columns)}
FROM {table_ref}
WHERE
    `{timestamp_field}` >= TIMESTAMP '{start_str}'
    AND `{timestamp_field}` <= TIMESTAMP '{end_str}'
        """.strip()

        return MaxComputeRetrievalJob(
            query=query,
            store_config=store_cfg,
            full_feature_names=False,
        )

    @staticmethod
    def write_logged_features(
        config: RepoConfig,
        data: Union[pa.Table, Path],
        source: LoggingSource,
        logging_config: LoggingConfig,
        registry: BaseRegistry,
    ) -> None:
        raise NotImplementedError(
            "Feature logging is not yet implemented for MaxComputeOfflineStore"
        )


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

def _col_list(columns: List[str]) -> str:
    """Return a safe backtick-quoted SQL column list."""
    return ", ".join(f"`{c}`" for c in columns)


def _build_point_in_time_query(
    config: RepoConfig,
    feature_views: List[FeatureView],
    feature_refs: List[str],
    entity_table_name: str,
    full_feature_names: bool,
) -> str:
    """
    Build a point-in-time correct SQL query using MaxCompute SQL dialect.

    For each feature view, LEFT JOIN the feature table onto the entity table
    using a sub-query that picks the latest row whose event_timestamp <=
    entity event_timestamp (and within the TTL window).
    """
    from feast_maxcompute.maxcompute_source import MaxComputeSource

    # Collect requested features per view
    fv_feature_map: Dict[str, List[str]] = {}
    for ref in feature_refs:
        view_name, feat_name = ref.split(":")
        fv_feature_map.setdefault(view_name, []).append(feat_name)

    join_parts: List[str] = []
    select_parts: List[str] = ["`entity_df`.*"]

    for fv in feature_views:
        if fv.name not in fv_feature_map:
            continue

        features = fv_feature_map[fv.name]
        assert isinstance(fv.batch_source, MaxComputeSource), (
            f"FeatureView '{fv.name}' must use a MaxComputeSource"
        )
        source: MaxComputeSource = fv.batch_source

        # For PIT queries there is no fixed date range, so we pass None —
        # static partitions will still be injected, date_str partitions are skipped.
        table_ref = source.get_table_query_string()

        ts_col = source.timestamp_field
        ts_col_event = "event_timestamp"  # standardize the timestamp column name for the PIT join
        join_keys = fv.join_keys  # list[str] in all recent Feast versions
        ttl_seconds = int(fv.ttl.total_seconds()) if fv.ttl else None
        alias = f"_fv_{fv.name}"

        # Join key conditions
        join_conditions = " AND ".join(
            f"`entity_df`.`{jk}` = `{alias}`.`{jk}`" for jk in join_keys
        )

        # Point-in-time condition
        pit_condition = (
            f"`{alias}`.`{ts_col}` <= `entity_df`.`{ts_col_event}`"
        )
        if ttl_seconds:
            pit_condition += (
                f" AND `{alias}`.`{ts_col}` >= "
                f"DATEADD(`entity_df`.`{ts_col_event}`, -{ttl_seconds}, 'ss')"
            )

        feature_col_list = _col_list(join_keys + [ts_col] + features)

        sub_query = (
            f"(SELECT {feature_col_list}, "
            f"ROW_NUMBER() OVER ("
            f"PARTITION BY {_col_list(join_keys)} "
            f"ORDER BY `{ts_col}` DESC"
            f") AS _rn FROM {table_ref})"
        )

        join_sql = (
            f"LEFT JOIN {sub_query} AS `{alias}`\n"
            f"    ON {join_conditions}\n"
            f"    AND {pit_condition}\n"
            f"    AND `{alias}`._rn = 1"
        )
        join_parts.append(join_sql)

        for feat in features:
            col_alias = f"{fv.name}__{feat}" if full_feature_names else feat
            select_parts.append(f"`{alias}`.`{feat}` AS `{col_alias}`")

    select_clause = ",\n    ".join(select_parts)
    joins_clause = "\n".join(join_parts)

    return f"""
SELECT
    {select_clause}
FROM `{entity_table_name}` AS `entity_df`
{joins_clause}
    """.strip()


# ---------------------------------------------------------------------------
# Entity DataFrame upload helper
# ---------------------------------------------------------------------------

def _upload_entity_df(
    odps: Any,
    store_cfg: MaxComputeOfflineStoreConfig,
    entity_df: Union[pd.DataFrame, str],
) -> str:
    """
    Upload entity_df to a temporary MaxCompute table and return its name.
    - pd.DataFrame → uploads via pyodps write API
    - str          → treated as an existing table name, returned as-is
    """
    if isinstance(entity_df, str):
        return entity_df

    if not isinstance(entity_df, pd.DataFrame):
        raise TypeError(
            f"entity_df must be a pd.DataFrame or str, got {type(entity_df)}"
        )

    uid = hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:8]
    temp_table = f"{store_cfg.temp_table_prefix}{uid}"

    logger.info(
        "Uploading entity_df (%d rows) to temporary MaxCompute table: %s",
        len(entity_df),
        temp_table,
    )

    ddl_columns = _pandas_dtypes_to_mc_ddl(entity_df)
    odps.execute_sql(
        f"CREATE TABLE IF NOT EXISTS `{temp_table}` ({ddl_columns}) LIFECYCLE 1"
    )

    mc_table = odps.get_table(temp_table)
    with mc_table.open_writer() as writer:
        writer.write(entity_df.values.tolist())

    return temp_table


def _pandas_dtypes_to_mc_ddl(df: pd.DataFrame) -> str:
    """Build a MaxCompute DDL column list from a pandas DataFrame."""
    dtype_map = {
        "int8": "TINYINT",
        "int16": "SMALLINT",
        "int32": "INT",
        "int64": "BIGINT",
        "float32": "FLOAT",
        "float64": "DOUBLE",
        "bool": "BOOLEAN",
        "object": "STRING",
        "datetime64[ns]": "TIMESTAMP",
        "datetime64[us]": "TIMESTAMP",
        "datetime64[ns, UTC]": "TIMESTAMP",
        "datetime64[us, UTC]": "TIMESTAMP",
    }
    parts = []
    for col, dtype in df.dtypes.items():
        mc_type = dtype_map.get(str(dtype), "STRING")
        parts.append(f"`{col}` {mc_type}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Retrieval metadata helper
# ---------------------------------------------------------------------------

def _get_retrieval_metadata(
    registry: BaseRegistry,
    feature_views: List[FeatureView],
    feature_refs: List[str],
    project: str,
    full_feature_names: bool,
) -> RetrievalMetadata:
    features_for_meta = []
    for ref in feature_refs:
        view_name, feat_name = ref.split(":")
        for fv in feature_views:
            if fv.name == view_name:
                col_name = (
                    f"{view_name}__{feat_name}" if full_feature_names else feat_name
                )
                features_for_meta.append(col_name)
                break

    return RetrievalMetadata(
        features=features_for_meta,
        keys=list(
            {jk for fv in feature_views for jk in fv.join_keys}
        ),
    )


# ---------------------------------------------------------------------------
# Range scan query builder (no entity_df / no PIT join)
# ---------------------------------------------------------------------------

def _build_range_scan_query(
    config: RepoConfig,
    feature_views: List[FeatureView],
    feature_refs: List[str],
    start_date: datetime,
    end_date: datetime,
    full_feature_names: bool,
) -> str:
    """
    Build a UNION ALL query that pulls every row across all requested feature
    views within [start_date, end_date]. No entity filtering, no PIT join —
    useful for training data exports when you want all available data.

    Each feature view contributes one SELECT block. Columns not present in a
    given view are filled with NULL so the UNION schema stays consistent.
    """
    from feast_maxcompute.maxcompute_source import MaxComputeSource

    # Collect all unique columns across all views for a consistent schema
    fv_feature_map: Dict[str, List[str]] = {}
    for ref in feature_refs:
        view_name, feat_name = ref.split(":")
        fv_feature_map.setdefault(view_name, []).append(feat_name)

    start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
    end_str   = end_date.strftime("%Y-%m-%d %H:%M:%S")

    blocks: List[str] = []

    for fv in feature_views:
        if fv.name not in fv_feature_map:
            continue

        features = fv_feature_map[fv.name]
        assert isinstance(fv.batch_source, MaxComputeSource), (
            f"FeatureView '{fv.name}' must use a MaxComputeSource"
        )
        source: MaxComputeSource = fv.batch_source
        ts_col = source.timestamp_field

        table_ref = source.get_table_query_string()

        join_keys = fv.join_keys
        all_cols  = join_keys + [ts_col] + features

        # Feature columns with optional view prefix
        feat_selects = []
        for feat in features:
            alias = f"{fv.name}__{feat}" if full_feature_names else feat
            feat_selects.append(f"`{feat}` AS `{alias}`")

        select_cols = (
            [f"`{c}`" for c in join_keys]
            + [f"`{ts_col}` AS `event_timestamp`"]
            + feat_selects
        )

        block = (
            f"SELECT {', '.join(select_cols)}\n"
            f"FROM {table_ref}\n"
            f"WHERE `{ts_col}` >= TIMESTAMP '{start_str}'\n"
            f"  AND `{ts_col}` <=  TIMESTAMP '{end_str}'"
        )
        blocks.append(block)

    if not blocks:
        raise ValueError(
            "No matching feature views found for the requested feature refs."
        )

    return "\nUNION ALL\n".join(blocks)


def _ensure_utc(dt: datetime) -> datetime:
    """Attach UTC timezone to a naive datetime; leave aware datetimes unchanged."""
    from datetime import timezone
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _cast_timestamp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timestamp-like columns are proper datetime objects."""
    for col in df.columns:
        if "timestamp" in col.lower() or col in ("event_timestamp", "created"):
            if df[col].dtype == object:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df