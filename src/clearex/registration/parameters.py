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


global_deform_kwargs = {
    # (1) pyramid & smoothing — Registration performed at this down-sampled resolution.
    "alignment_spacing": 1.0,
    # The grid size in physical units (microns) which to cast the results to.
    "control_point_spacing": 16.0,
    # Multiscale - Multiplied with control_point_spacing in pyramidal fashion.
    "control_point_levels": [1],
    # Downsampling scale levels at which to optimize. This sets up multiscale alignment. For example (4, 2, 1)
    # will repeat the registration optimization three times. First, it will register at 4x downsampling of the data along
    # all axes. Then at 2x downsampling, then finally at the full given resolution.
    #'shrink_factors': (1,),
    # Sigma of Gaussian used to smooth each scale level image. Must be same length as `shrink_factors`
    "smooth_sigmas": (0.0,),
    # (2) metric & sampling
    # 'MS':    Mean-Squares for smooth deforms
    # 'ANC':   AntsNiehborhoodCorrelation
    # 'C':     Correlation
    # 'D':     Demons
    # 'JHMI':  JointHistogramMutualInformation
    # 'MMI':   MattesMutualInformation. Default value.
    "metric": "MMI",
    # How image intensities are sampled in space during metric calculation
    # 'NONE':     All voxels are used, values from voxel centers. Default value.
    # 'REGULAR':  Regular spacing between samples, small random perturbation from voxel centers
    # 'RANDOM':   Sample positions are totally random
    "sampling": "NONE",
    # (4) optimizer & regularization
    # 'A':        Amoeba
    # 'CGLS':     ConjugateGradientLineSearch
    # 'E':        Exhaustive
    # 'GD':       GradientDescent
    # 'GDLS':     GradientDescentLineSearch
    # 'LBFGS2':   LimitedMemoryBroydenFletcherGoldfarbShannon w/o bounds
    # 'LBFGSB':   LimitedMemoryBroydenFletcherGoldfardShannon w/ simple bounds
    # 'OPOE':     OnePlueOneEvolutionary
    # 'P':        Powell
    # 'RSGD':     RegularStepGradientDescent. Default value.
    "optimizer": "RSGD",
    # (5) interpolation
    # '0':    NearestNeighbor,
    # '1':    Linear,
    # 'CBS':   CubicBSpline,
    # 'G':    Gaussian,
    # 'LG':   LabelGaussian,
    # 'HWS':  HammingWindowedSinc,
    # 'CWS':  CosineWindowedSinc,
    # 'WWS':  WelchWindowedSinc,
    # 'LWS':  LanczosWindowedSinc,
    # 'BWS':  BlackmanWindowedSinc,
    "interpolator": "1",
}

if (
    global_deform_kwargs["sampling"] == "RANDOM"
    or global_deform_kwargs["sampling"] == "REGULAR"
):
    # Specify the percentage of voxels to use as a fraction between 0 and 1.
    global_deform_kwargs["sampling_percentage"] = 0.2


if global_deform_kwargs["optimizer"] == "LBFGS2":
    global_deform_kwargs["optimizer_args"] = {
        "lineSearchAccuracy": 0.9,
        "numberOfIterations": 100,
    }

if global_deform_kwargs["optimizer"] == "RSGD":
    global_deform_kwargs["optimizer_args"] = {
        "learningRate": 1.5,
        "minStep": 0.0,
        "numberOfIterations": 250,
    }
