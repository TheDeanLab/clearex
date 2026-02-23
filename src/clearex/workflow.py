#  Copyright (c) 2021-2025  The University of Texas Southwestern Medical Center.
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted for academic and research use only (subject to the
#  limitations in the disclaimer below) provided that the following conditions are met:
#       * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#       * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#       * Neither the name of the copyright holders nor the names of its
#       contributors may be used to endorse or promote products derived from this
#       software without specific prior written permission.
#  NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
#  THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
#  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
#  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
#  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

from dataclasses import dataclass
from typing import Optional, Tuple, Union


ChunkSpec = Optional[Union[int, Tuple[int, ...]]]


@dataclass
class WorkflowConfig:
    """Runtime workflow options shared by GUI and headless entrypoints.

    Attributes
    ----------
    file : str, optional
        Input image path for processing.
    prefer_dask : bool
        Whether to open data using lazy Dask-backed arrays when supported.
    chunks : int or tuple of int, optional
        Chunking configuration used for Dask reads.
    deconvolution : bool
        Flag indicating whether deconvolution workflow should run.
    particle_detection : bool
        Flag indicating whether particle detection workflow should run.
    registration : bool
        Flag indicating whether registration workflow should run.
    visualization : bool
        Flag indicating whether visualization workflow should run.
    """

    file: Optional[str] = None
    prefer_dask: bool = True
    chunks: ChunkSpec = None
    deconvolution: bool = False
    particle_detection: bool = False
    registration: bool = False
    visualization: bool = False

    def has_analysis_selection(self) -> bool:
        """Return whether at least one analysis operation is selected.

        Returns
        -------
        bool
            ``True`` if any analysis flag is enabled, otherwise ``False``.
        """
        return any(
            (
                self.deconvolution,
                self.particle_detection,
                self.registration,
                self.visualization,
            )
        )


def parse_chunks(chunks: Optional[str]) -> ChunkSpec:
    """Parse chunk spec from CLI/GUI text.

    Parameters
    ----------
    chunks : str, optional
        A single integer (e.g., ``"256"``) or comma-separated tuple
        (e.g., ``"1,256,256"``). Empty strings are treated as ``None``.

    Returns
    -------
    Optional[int | Tuple[int, ...]]
        Parsed chunk specification or ``None``.

    Raises
    ------
    ValueError
        If ``chunks`` cannot be parsed as integers or contains non-positive
        values.
    """
    if chunks is None:
        return None

    text = chunks.strip()
    if not text:
        return None

    if "," not in text:
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(
                "Chunks must be a positive integer or comma-separated integers."
            ) from exc
        if value <= 0:
            raise ValueError("Chunk size must be greater than zero.")
        return value

    try:
        parts = tuple(int(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise ValueError(
            "Chunks must be a positive integer or comma-separated integers."
        ) from exc
    if not parts:
        return None
    if any(value <= 0 for value in parts):
        raise ValueError("Chunk sizes must be greater than zero.")
    return parts


def format_chunks(chunks: ChunkSpec) -> str:
    """Format a chunk specification for display.

    Parameters
    ----------
    chunks : int or tuple of int, optional
        Chunk specification to serialize.

    Returns
    -------
    str
        Empty string when ``chunks`` is ``None``. Otherwise returns a single
        integer or comma-separated integer list.
    """
    if chunks is None:
        return ""
    if isinstance(chunks, tuple):
        return ",".join(str(part) for part in chunks)
    return str(chunks)
