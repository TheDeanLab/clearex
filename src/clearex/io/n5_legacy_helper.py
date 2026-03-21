"""Legacy N5 materialization entrypoint for zarr2-compatible Python runtimes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from clearex.io.experiment import (
    create_dask_client,
    load_navigate_experiment,
    materialize_experiment_data_store,
)


def _parse_chunks(text: str) -> tuple[int, int, int, int, int, int]:
    values = tuple(int(part.strip()) for part in str(text).split(",") if part.strip())
    if len(values) != 6:
        raise ValueError("chunks must define six comma-separated integers.")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize an N5 source into a legacy ClearEx Zarr v2 store."
    )
    parser.add_argument("--experiment-path", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--output-store", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--pyramid-factors", required=True)
    parser.add_argument("--scheduler-address", default=None)
    args = parser.parse_args()

    chunks = _parse_chunks(str(args.chunks))
    pyramid_factors_raw = json.loads(str(args.pyramid_factors))
    pyramid_factors = tuple(
        tuple(int(value) for value in axis_levels) for axis_levels in pyramid_factors_raw
    )
    if len(pyramid_factors) != 6:
        raise ValueError("pyramid_factors must define six axis entries.")

    os.environ["CLEAREX_LEGACY_N5_ACTIVE"] = "1"
    os.environ["CLEAREX_TARGET_ZARR_FORMAT"] = "2"
    os.environ["CLEAREX_OVERRIDE_ANALYSIS_STORE_PATH"] = str(
        Path(args.output_store).expanduser().resolve()
    )

    helper_client = None
    try:
        scheduler_address = str(args.scheduler_address or "").strip()
        if scheduler_address:
            helper_client = create_dask_client(
                scheduler_address=scheduler_address
            )

        experiment = load_navigate_experiment(Path(args.experiment_path))
        materialize_experiment_data_store(
            experiment=experiment,
            source_path=Path(args.source_path),
            chunks=chunks,
            pyramid_factors=pyramid_factors,
            client=helper_client,
            force_rebuild=True,
        )
    finally:
        if helper_client is not None:
            try:
                helper_client.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
