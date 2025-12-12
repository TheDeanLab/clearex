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

# Standard library imports
from typing import Tuple, List

# Third party imports
import napari

# Local imports


def create_viewer(*, show: bool = True) -> napari.Viewer:
    """
    Create and return a new napari viewer instance.

    Returns
    -------
    napari.Viewer
        A new napari viewer instance.
    """
    viewer = napari.Viewer(ndisplay=3, show=show)
    return viewer


def update_viewer(
    viewer: napari.Viewer,
    angles: Tuple[float, float, float],
    zoom: float,
    center: Tuple[float, float, float],
    time: int,
) -> None:
    """
    Update the napari viewer's camera perspective and time step.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance to update.
    angles : Tuple[float, float, float]
        Camera rotation angles (azimuth, elevation, roll) in degrees.
    zoom : float
        Camera zoom level.
    center : Tuple[float, float, float]
        Camera center point coordinates (z, y, x).
    time : int
        Time step index to set in the viewer dimensions.

    Returns
    -------
    None
    """
    viewer.camera.angles = angles
    viewer.camera.zoom = zoom
    viewer.camera.center = center
    step = list(viewer.dims.current_step)
    step[0] = time
    viewer.dims.current_step = tuple(step)


def get_viewer_perspective(
    viewer: napari.Viewer,
) -> Tuple[
    Tuple[float, float, float], float, Tuple[float, float, float], List[int], int
]:
    """
    Get the current camera perspective and time step from a napari viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance to query.

    Returns
    -------
    angle : Tuple[float, float, float]
        Camera rotation angles (azimuth, elevation, roll) in degrees.
    zoom : float
        Camera zoom level.
    center : Tuple[float, float, float]
        Camera center point coordinates (z, y, x).
    step : List[int]
        Current dimension step indices.
    time : int
        Time step index (first element of step).
    """
    angle = viewer.camera.angles
    zoom = viewer.camera.zoom
    center = viewer.camera.center
    step = list(viewer.dims.current_step)
    time = step[0]
    return angle, zoom, center, step, time
