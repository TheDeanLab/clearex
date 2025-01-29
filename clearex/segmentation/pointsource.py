
import numpy as np

def remove_close_blobs(blobs, min_dist):
    """ Remove blobs that are too close to each other.

    Takes into account the sigma value to calculate the minimum distance.

    Parameters
    ----------
    blobs : np.ndarray
        An Nx4 array of blobs where the columns are [z, y, x, sigma].
    min_dist : float
        The minimum distance between blobs.

    Returns
    -------
    np.ndarray
        An Nx4 array of blobs where the columns are [z, y, x, sigma].
    """
    final_blobs = []
    for b in blobs:

        # Determine radius from sigma
        if len(b) == 4:
            # Isotropic sigma: [z, y, x, sigma]
            radius_b = b[3]
        elif len(b) == 6:
            # Anisotropic sigma: [z, y, x, z_sigma, y_sigma, x_sigma]
            z_sigma, y_sigma, x_sigma = b[3], b[4], b[5]
            radius_b = np.mean([z_sigma, y_sigma, x_sigma])
        else:
            raise ValueError("Expected blobs to have shape (N,4) or (N,6). Got length {}.".format(len(b)))

        center_b = b[:3]  # (z, y, x)

        # Check if this blob center is far enough from all previously kept centers
        too_close = False
        for k in final_blobs:
            center_k = k[:3]

            if len(k) == 4:
                radius_k = k[3] * min_dist
            else:
                z_sigma_k = k[3] * min_dist
                y_sigma_k = k[4] * min_dist
                x_sigma_k = k[5] * min_dist

                # Take into account anisotropic sigmas for non-spherical search
                

                radius_k = np.mean([z_sigma_k, y_sigma_k, x_sigma_k])

            dist = np.linalg.norm(center_b - center_k)

            # "Too close" if distance < sum_of_radii + min_dist
            if dist < (radius_b + radius_k + min_dist):
                too_close = True
                break

        if not too_close:
            final_blobs.append(b)
    return np.array(final_blobs)

