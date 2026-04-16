# Fusion Gain Compensation Performance Design

Date: 2026-04-16
Status: Approved design for planning
Scope: `src/clearex/registration/pipeline.py`, `src/clearex/registration/README.md`, fusion tests, fusion provenance/metadata

## Problem

Large production fusion runs on shared-filesystem HPC storage spend most of
their wall clock in the `gain_compensated_feather` intensity-correction
prepass before fused chunk rendering begins.

The motivating example is:

- dataset: `/archive/bioinformatics/Danuser_lab/Dean/dean/2026-04-07-liver`
- canonical store: `data_store.ome.zarr`
- fused output shape around `(1, 1, 4, 590, 30995, 14319)`
- output chunking around `(1, 1, 1, 256, 256, 256)`

Observed runs on this store show:

- `2026-04-15 21:26:20` to `2026-04-16 02:15:55`: fusion sat between `12%`
  and `20%` while estimating intensity corrections for
  `gain_compensated_feather`
- once chunk rendering began, the `20% -> 100%` portion completed in about
  `34 minutes`
- the current `2026-04-16 13:05:12` run is again stuck at
  `[fusion] 12% - Estimating fusion intensity corrections`
- current GUI state uses `blend_overlap_zyx=[400, 400, 400]`, which further
  enlarges the overlap crops used by the prepass

This means the primary optimization target is not the fused chunk loop. It is
the overlap intensity-fit prepass used by `gain_compensated_feather`.

## Goals

- Preserve `gain_compensated_feather` as a production-quality fusion mode.
- Reduce first-run wall clock for the intensity-fit prepass on very large
  volumes over network storage.
- Make repeated reruns on the same registration result substantially cheaper.
- Keep final fused image generation at full resolution.
- Keep new persisted artifacts small and metadata-like, not image-like.
- Improve operator visibility so long intensity-fit phases no longer appear
  hung at one progress value.

## Non-Goals

- Bitwise-identical gain/offset values relative to the current implementation.
- Changing the public fusion output contract.
- Replacing `gain_compensated_feather` with `feather` as the default.
- Persisting large overlap crops, temporary volumes, or fusion-side pyramids.

## Existing Bottleneck

Today `_estimate_pairwise_overlap_intensity(...)` does the following per edge:

1. Build a world-space overlap crop from the registration overlap bounding box.
2. Read the required source subvolumes from both tiles.
3. Resample both source subvolumes into a full overlap world grid.
4. Fit `fixed ~= scale * moving + offset` using trimmed nonzero finite voxels.

This is expensive on shared filesystems because:

- the overlap world grid can become very large
- both tiles are read separately for each edge fit
- both subvolumes are resampled before the fit
- reruns repay the same cost even when registration geometry is unchanged

## Approaches Considered

### 1. Sampled plus cached edge-intensity fits

Keep `gain_compensated_feather`, but estimate edge gains/offsets from a bounded
estimation grid, validate the coarse fit against sparse full-resolution
samples, and persist the resulting per-edge fit metadata for reuse on reruns.

Pros:

- improves first-run latency
- improves rerun latency
- stores only small metadata artifacts
- preserves full-resolution final fusion

Cons:

- adds cache invalidation logic
- adds a small policy surface for estimation scale and validation

### 2. Compute-only coarse estimation

Only change the estimator to use bounded downsampled grids, with no persisted
reuse.

Pros:

- simpler implementation
- lower metadata complexity

Cons:

- reruns still pay most of the prepass cost
- weaker operational benefit for tuning workflows

### 3. Registration-coupled overlap summaries

Persist richer overlap summaries during registration so fusion can reuse them
later.

Pros:

- strongest rerun story

Cons:

- tighter coupling between registration and fusion
- more state to keep coherent
- larger surface area than needed for the immediate bottleneck

## Recommendation

Use approach 1.

Implement a bounded, scale-aware intensity-fit prepass for
`gain_compensated_feather`, then persist small per-edge fit metadata under the
fusion auxiliary tree for reuse when registration geometry and estimation
policy are unchanged.

## Proposed Execution Model

### Prepass Flow

For each timepoint and registration edge:

1. Derive the nominal overlap crop exactly as today.
2. Estimate the full-resolution overlap voxel count.
3. Choose an estimation scale from `{1, 2, 4}` in uniform Z/Y/X downsampling.
4. Resample fixed and moving overlap crops onto the estimation grid only.
5. Fit linear gain/offset on the estimation grid.
6. Validate the coarse fit against sparse full-resolution samples from the same
   overlap.
7. If validation passes, accept the fit.
8. If validation fails, retry at the next denser scale.
9. If scale `1` still fails validation or lacks enough valid samples, record an
   identity gain/offset for that edge and continue.

The final fused chunk loop remains unchanged and still renders the final fused
volume at full resolution.

### Scale Selection

Add `intensity_estimation_scale_mode` with these values:

- `auto` default
- `full_res`
- `downsample_2x`
- `downsample_4x`

Add `max_intensity_fit_voxels` with default `8_000_000`.

Scale selection rules:

- `full_res`: always use scale `1`
- `downsample_2x`: always start at scale `2`, then validate, then escalate to
  `1` on failure
- `downsample_4x`: always start at scale `4`, then validate, then escalate to
  `2`, then `1`
- `auto`: choose the coarsest scale in `{4, 2, 1}` whose estimated overlap
  voxel count is less than or equal to `max_intensity_fit_voxels`; if none
  fits, start at `4`

This gives the requested `2x` and `4x` cheaper-compute options while keeping a
quality-preserving fallback path.

### Validation

Add these validation controls:

- `full_res_validation_samples` default `50_000`
- `max_intensity_validation_rmse_fraction` default `0.05`
- `min_intensity_validation_correlation` default `0.95`

Validation procedure:

1. Draw a deterministic sparse sample from full-resolution overlap voxels after
   existing finite/nonzero masking.
2. Apply the estimated gain/offset to the sampled moving intensities.
3. Compute:
   - Pearson correlation between corrected moving and fixed samples
   - RMSE normalized by the fixed-sample intensity range
4. Accept the fit when:
   - correlation is at least `min_intensity_validation_correlation`
   - normalized RMSE is at most `max_intensity_validation_rmse_fraction`

If the fit fails validation:

- escalate to a denser scale when one exists
- otherwise fall back to scale `1`
- if the scale `1` fit still lacks enough valid samples, use identity
  gain/offset for that edge and record the failure status

## Reuse Model

Persist reusable fit metadata under:

- `clearex/results/fusion/latest/intensity_fit_edges`

This is a small metadata group, not an image store.

### Proposed Stored Fields

Store one array per field for consistency with existing Zarr usage:

- `edge_pairs_pe2`: `int32`, shape `(edge_count, 2)`
- `gains_te`: `float32`, shape `(t_count, edge_count)`
- `offsets_te`: `float32`, shape `(t_count, edge_count)`
- `status_te`: `uint8`, shape `(t_count, edge_count)`
- `estimation_scale_te3`: `uint8`, shape `(t_count, edge_count, 3)`
- `fit_sample_count_te`: `int32`, shape `(t_count, edge_count)`
- `validation_sample_count_te`: `int32`, shape `(t_count, edge_count)`
- `validation_corr_te`: `float32`, shape `(t_count, edge_count)`
- `validation_nrmse_te`: `float32`, shape `(t_count, edge_count)`

Group attrs include:

- `source_component`
- `registration_component`
- `affines_component`
- `transformed_bboxes_component`
- `blend_mode`
- `gain_clip_range`
- `intensity_estimation_scale_mode`
- `max_intensity_fit_voxels`
- `full_res_validation_samples`
- `max_intensity_validation_rmse_fraction`
- `min_intensity_validation_correlation`
- `cache_policy_version`
- `geometry_sha256`

### Reuse Key

Reuse cached edge fits only when all of the following match:

- `source_component`
- `registration_component`
- `affines_component`
- `transformed_bboxes_component`
- `blend_mode`
- `gain_clip_range`
- all estimation-policy parameters
- `geometry_sha256`

`geometry_sha256` is computed from the content of:

- `affines_tpx44`
- `transformed_bboxes_tpx6`
- `edges_pe2`

These arrays are small relative to source volumes, so hashing them is cheap and
avoids relying on brittle filesystem timestamps.

Cache storage remains `latest_only`, matching existing fusion metadata
semantics. Provenance remains append-only and authoritative for run history.

Implementation note:

- when a fusion rerun replaces `clearex/results/fusion/latest`, the runtime
  must inspect and load any reusable `intensity_fit_edges` metadata before the
  old latest group is cleared, then write accepted cache metadata back into the
  new latest group

## Progress And Operator Feedback

Replace the current single long `12%` intensity-correction phase with granular
edge progress:

- `12%`: beginning intensity-fit prepass
- `12% -> 20%`: proportional to completed edge fits across all timepoints
- progress message should include completed edges, total edges, and the current
  accepted estimation scale for the latest completed edge

Example:

- `[fusion] 16% - Intensity fits 7/24 edges (latest scale=4x, fallback=no)`

This does not change throughput, but it removes the current “appears stuck”
operator experience.

## Error Handling

- If an overlap crop has too few valid samples, record failure and use identity
  gain/offset for that edge.
- If cache metadata is missing or invalid, recompute without failing the whole
  fusion run.
- If a coarse fit is numerically unstable or fails validation, escalate to a
  denser scale automatically.
- If all scales fail, keep
  identity gain/offset and continue.

Fusion should not fail only because one edge cannot support a robust intensity
fit.

## Data Quality Expectations

This design preserves “same-quality” output in the practical sense:

- final fusion still uses `gain_compensated_feather`
- final fused chunks still render at full resolution
- coarse estimation is accepted only after passing full-resolution validation
- problematic edges automatically escalate to denser estimation

The design does not guarantee bit-identical gain/offset values relative to the
current implementation. The acceptance target is visually equivalent seam
quality with materially lower latency.

## Validation Plan

### Unit Tests

- scale selection for `auto`, `downsample_2x`, `downsample_4x`, and `full_res`
- cache reuse when geometry and policy match
- cache invalidation when geometry hash or policy attrs change
- fallback ordering from `4x -> 2x -> 1x`
- identity fallback when sample counts remain insufficient

### Quality Tests

- synthetic overlap pairs with known gain/offset should recover comparable fits
  at `1x`, `2x`, and `4x`
- validation should reject intentionally bad coarse fits
- accepted coarse fits should stay within bounded metric drift relative to the
  full-resolution estimator

### Integration Tests

- rerunning fusion against unchanged registration reuses cached edge fits and
  skips recomputation
- `gain_compensated_feather` still improves known intensity-mismatch seams
- progress updates advance during the intensity-fit prepass instead of staying
  flat at one percent

## Documentation Updates Required

Update:

- `src/clearex/registration/README.md`
- relevant runtime docs if fusion parameter semantics change

Documentation must explain:

- new gain-compensation estimation controls
- `auto` scale selection
- rerun reuse behavior
- the fact that downsampling affects only the gain-estimation prepass, not the
  final fused image resolution

## Implementation Boundaries

Primary code changes are expected in:

- `src/clearex/registration/pipeline.py`
- `tests/registration/test_pipeline.py`
- `src/clearex/registration/README.md`

No changes are required to the public fusion image contract under
`results/fusion/latest`.

## Rollout

1. Add the new estimation-policy parameters with defaults that preserve current
   behavior quality expectations.
2. Implement coarse estimation plus validation fallback.
3. Add persisted edge-fit cache and invalidation logic.
4. Add granular progress reporting for the prepass.
5. Update tests and docs.

## Decision Summary

ClearEx should keep `gain_compensated_feather`, but make it practical on large
shared-filesystem HPC workloads by:

- bounding overlap intensity estimation with `1x/2x/4x` estimation scales
- validating coarse fits against sparse full-resolution samples
- persisting small per-edge fit metadata for cheap reruns
- keeping final fused chunk rendering at full resolution
