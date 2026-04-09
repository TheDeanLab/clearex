.. _analysis-workflow-volume-export:

Analysis Workflow: Volume Export
================================

Implemented behavior
--------------------

This analysis path exports one selected image-producing ClearEx source component
from a canonical OME-Zarr store into either an in-store public OME-Zarr result
or in-store OME-TIFF artifacts.

``volume_export`` is part of the visualization workflow family, but it exports
resolved runtime data rather than viewer metadata. The source can be any
previously materialized image-producing component, including the canonical
source data plus flatfield, deconvolution, shear transform, fusion, or other
analysis outputs that resolve to canonical internal 6D arrays in
``(t, p, c, z, y, x)`` order.

Default analysis parameter dictionary
-------------------------------------

``WorkflowConfig`` carries ``analysis_parameters``, keyed by analysis method,
with defaults in ``DEFAULT_ANALYSIS_OPERATION_PARAMETERS``.

All operation dictionaries include:

- ``execution_order`` (int): defines run order when multiple routines are
  selected. ``volume_export`` defaults to ``12`` so it runs after
  ``compile_movie`` and before ``mip_export``.
- ``input_source`` (str): logical source alias or explicit internal component
  path. The default is ``data``.
- ``force_rerun`` (bool): bypasses provenance-based skip logic and reruns the
  export even when a matching latest result already exists. The default is
  ``False``.

The ``volume_export`` defaults include:

- ``chunk_basis`` (str, default ``"3d"``): normalized execution-compatibility
  field inherited from other volume analyses. ``volume_export`` pins this to
  3D chunks and does not currently expose alternate chunk bases.
- ``detect_2d_per_slice`` (bool, default ``False``): compatibility field
  inherited from chunked detection workflows. It is normalized for consistency
  but not used by ``volume_export``.
- ``use_map_overlap`` (bool, default ``False``): compatibility field inherited
  from chunk-overlap planning. ``volume_export`` keeps overlap masking off.
- ``overlap_zyx`` (list[int], default ``[0, 0, 0]``): compatibility field for
  overlap-aware chunk planning. ``volume_export`` does not use overlap padding.
- ``memory_overhead_factor`` (float, default ``1.0``): normalized scheduler
  hint carried through the workflow payload. The current export implementation
  runs eagerly and does not consume this field directly, but it remains part of
  the persisted parameter set.
- ``export_scope`` (str, default ``"current_selection"``): chooses between
  exporting one selected ``(t, p, c)`` volume or exporting all available
  indices. Accepted values are ``current_selection`` and ``all_indices``.
- ``t_index`` (int, default ``0``): time index used only when
  ``export_scope=current_selection``. Negative values are normalized to ``0``.
- ``p_index`` (int, default ``0``): position index used only when
  ``export_scope=current_selection``. Negative values are normalized to ``0``.
- ``c_index`` (int, default ``0``): channel index used only when
  ``export_scope=current_selection``. Negative values are normalized to ``0``.
- ``resolution_level`` (int, default ``0``): source-adjacent pyramid level to
  export. ``0`` uses the base source component. Higher levels reuse existing
  display/source-adjacent levels when present and generate missing deeper levels
  on demand during export.
- ``export_format`` (str, default ``"ome-zarr"``): chooses between public
  in-store OME-Zarr publication and in-store OME-TIFF artifacts. Accepted
  normalized values are ``ome-zarr`` and ``ome-tiff``; aliases such as
  ``zarr`` and ``tiff`` normalize to those canonical names.
- ``tiff_file_layout`` (str, default ``"single_file"``): TIFF-only policy that
  chooses one BigTIFF/OME-TIFF file versus one file per exported volume.
  Accepted values are ``single_file`` and ``per_volume_files``.

Execution sequence and upstream input behavior
----------------------------------------------

- Runtime resolves ``input_source`` through the shared analysis-source alias
  resolver.
- Common aliases include:

  - ``data`` -> ``clearex/runtime_cache/source/data``
  - ``flatfield`` -> ``clearex/runtime_cache/results/flatfield/latest/data``
  - ``deconvolution`` -> ``clearex/runtime_cache/results/deconvolution/latest/data``
  - ``shear_transform`` -> ``clearex/runtime_cache/results/shear_transform/latest/data``
  - ``fusion`` -> ``clearex/runtime_cache/results/fusion/latest/data``
  - ``usegment3d`` -> ``clearex/runtime_cache/results/usegment3d/latest/data``

- Explicit internal component paths are also supported when they resolve to a
  canonical 6D image array.
- If the requested source component is missing or is not a canonical 6D array,
  ``volume_export`` raises an input dependency error instead of silently
  falling back.

Export scope and resolution behavior
------------------------------------

- ``current_selection`` slices the resolved source to exactly one
  ``(t, p, c, z, y, x)`` payload with singleton leading axes.
- ``all_indices`` exports the full resolved ``(t, p, c, z, y, x)`` array for
  the selected source.
- ``resolution_level=0`` uses the base source component directly.
- Higher ``resolution_level`` values first reuse existing source-adjacent
  levels discovered for the selected source.
- If the requested level is not already materialized, ``volume_export`` builds
  the missing deeper level(s) before writing the export payload.
- Resolved voxel calibration is scaled to the selected level and persisted in
  metadata plus OME output headers.

Volume-export execution workflow
--------------------------------

1. Normalize the ``volume_export`` parameter block.
2. Resolve ``input_source`` to the canonical internal source component.
3. Resolve or generate the requested ``resolution_level`` component.
4. Validate the selected ``t_index``, ``p_index``, and ``c_index`` when
   ``export_scope=current_selection``.
5. Rewrite the export runtime-cache payload at
   ``clearex/runtime_cache/results/volume_export/latest/data``.
6. Materialize either:

   - a public OME image collection under ``results/volume_export/latest`` when
     ``export_format=ome-zarr``, or
   - OME-TIFF artifacts under
     ``clearex/results/volume_export/latest/files`` when
     ``export_format=ome-tiff``.

7. Rewrite latest export metadata under ``clearex/results/volume_export/latest``.
8. Register the latest-output reference in provenance.

Storage and output contract
---------------------------

The export always rewrites the latest runtime payload under:

- ``clearex/runtime_cache/results/volume_export/latest/data``

Latest metadata for the run is stored under:

- ``clearex/results/volume_export/latest``

The metadata group records:

- requested ``input_source``,
- resolved source and resolution components,
- ``export_scope``,
- ``resolution_level``,
- whether deeper levels were generated on demand,
- ``export_format``,
- ``tiff_file_layout``,
- selected shape metadata,
- resolved voxel size,
- emitted artifact paths.

OME-Zarr output contract
------------------------

- ``export_format=ome-zarr`` publishes a public OME image collection under
  ``results/volume_export/latest``.
- ``current_selection`` publishes one exported field at the selected
  resolution.
- ``all_indices`` publishes every available exported field from the resolved
  source.
- The public result reflects the selected export resolution; it does not
  publish a new multiscale pyramid for the export itself.

OME-TIFF layout policy
----------------------

- ``current_selection`` always writes one ``ZYX`` OME-TIFF/BigTIFF file named
  from the selected ``t/p/c`` indices.
- ``all_indices + single_file`` writes one OME-TIFF/BigTIFF file with one
  ``TCZYX`` series per exported position.
- ``all_indices + per_volume_files`` writes one ``ZYX`` OME-TIFF file per
  exported ``(t, p, c)`` volume.
- TIFF artifacts are stored inside the analysis store under
  ``clearex/results/volume_export/latest/files`` so they remain namespaced with
  the latest export metadata.
- TIFF metadata preserves resolved ``PhysicalSizeZ/Y/X`` calibration.

CLI and GUI surface
-------------------

- The CLI currently exposes the operation flag ``--volume-export``.
- Detailed parameter editing is currently done through the GUI or a programmatic
  ``WorkflowConfig.analysis_parameters["volume_export"]`` payload.
- The visualization GUI exposes controls for:

  - ``input_source``
  - ``export_scope``
  - ``t_index`` / ``p_index`` / ``c_index``
  - ``resolution_level``
  - ``export_format``
  - ``tiff_file_layout``

Provenance integration
----------------------

- ``volume_export`` registers its latest output reference under
  ``clearex/provenance/latest_outputs/volume_export``.
- Workflow provenance stores the normalized parameter block plus resolved export
  metadata such as resolved source component, selected scope, resolution, file
  layout, and artifact paths.
- OME-Zarr exports publish a public latest analysis collection; OME-TIFF
  exports clear any stale public ``results/volume_export/latest`` collection so
  the public store contract matches the selected export format.

Verification
------------

When this workflow changes, validation should cover:

- parameter normalization for all accepted scope/format/layout values,
- current-selection versus all-indices export behavior,
- on-demand generation of missing resolution levels,
- voxel-calibration preservation for OME-Zarr and OME-TIFF outputs,
- cleanup of stale TIFF artifacts or stale public OME collections,
- provenance latest-output registration.
