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

from PyPetaKit5D import XR_decon_data_wrapper


def richardson_lucy(
    data_paths: list[str],
    PSFPath: str,
    xy_pixel: float,
    z_pixel: float,
    channel_pattern: list[str],
) -> None:
    """
    Perform Richardson-Lucy deconvolution on image data.

    Uses the PyPetaKit5D library to perform deconvolution with the
    Optimal Wiener-Butterworth (OWM) Richardson-Lucy method.

    Parameters
    ----------
    data_paths : list[str]
        List of file paths to the image data to be deconvolved.
    PSFPath : str
        Path to the Point Spread Function (PSF) file.
    xy_pixel : float
        Pixel size in the XY plane in microns.
    z_pixel : float
        Pixel size in the Z direction (step size) in microns.
    channel_pattern : list[str]
        List of channel patterns to match for processing.

    Returns
    -------
    None
        Results are saved to a 'decon' subdirectory in the input data path.

    Notes
    -----
    The function uses the following default parameters:
    - RLMethod: 'omw' (Optimal Wiener-Butterworth)
    - wienerAlpha: 0.005
    - OTFCumThresh: 0.6
    - deconIter: 2
    - background: 110
    - Output is saved as 16-bit images.
    """
    params = {
        "channelPatterns": channel_pattern,
        "resultDirName": "decon",
        "xyPixelSize": xy_pixel,
        "dz": z_pixel,
        "dzPSF": z_pixel,
        "hannWindowounds": [0.8, 1],
        "reverse": False,
        "psfFullpaths": [PSFPath],
        "parseSettingFile": False,
        "RLMethod": "omw",
        "wienerAlpha": 0.005,
        "OTFCumThresh": 0.6,
        "skewed": False,
        "showOTFMasks": True,
        "edgeErosion": 0,
        "background": 110,
        "deconIter": 2,
        "save16bit": True,
        "zarrFile": False,
        "parseCluster": False,
        "largeFile": False,
        "GPUJob": False,
        "debug": False,
        "cpusPerTask": 60,
        "mccMode": True,
        "GPUConfigFile": "",
        "configFile": "",
        "overwrite": True,
    }
    XR_decon_data_wrapper(data_paths, **params)
