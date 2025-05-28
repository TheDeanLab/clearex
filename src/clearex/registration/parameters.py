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
