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

# Standard Library Imports
import os
import logging
import io
import contextlib
import shutil

# Third Party Imports
import typer
import tifffile
import numpy as np
import ants

import clearex.registration.linear
# Local Imports
import clearex.registration.linear as linear
import clearex.registration.nonlinear as nonlinear
from clearex import setup_logger, log_and_echo as log
from clearex.file_operations.tools import crop_overlapping_datasets
import clearex.registration.common


class Registration:
    """ A class to register data. """

    def __init__(self):
        #self.moving_path = typer.prompt("What directory is the moving data in?")
        self.moving_path = ("/archive/bioinformatics/Danuser_lab/Dean/dean/2025-06"
                               "-12-registration/moving")

        #: str: The path to the moving data.
        # self.reference_path = typer.prompt("What directory is the reference data in?")
        self.reference_path = ("/archive/bioinformatics/Danuser_lab/Dean/dean/2025-06"
                               "-12-registration/fixed")

        # str: The accuracy to one the registration at. Low or High
        self.accuracy = "dry run"

        # str: The base path to save the data to
        self.saving_path = os.path.join(self.moving_path, "registration")

        #: str: The path to save the linear registration results.
        self.linear_path = os.path.join(self.saving_path, "linear")

        #: str: The path to save the nonlinear registration results.
        self.nonlinear_path = os.path.join(self.saving_path, "nonlinear")

        #: str: The path to save the fixed reference data.
        self.fixed_path = os.path.join(self.saving_path, "fixed")

        # Make sure the directories exist.
        self.make_directories()

        #: logging.Logger: The logger for documenting registration steps.
        self.logger = setup_logger(name='registration', log_file=os.path.join(
            self.moving_path, "registration", "registration.log"), level=logging.DEBUG)

        # Log paths.
        log(self, message="*** Initializing New Registration Runtime ***")
        log(self, message=f"Moving path: {self.moving_path}")
        log(self, message=f"Registration path: {self.reference_path}")

        #: str: The channel to use for registration.
        self.channel = "CH01"
        # self.channel = None

        #: np.ndarray: The reference image data.
        self.reference_data = None

        #: str: The filename of the reference data.
        self.reference_data_filename = None

        #: list[str]: A list of other filenames in the reference directory.
        self.reference_data_other_filenames = None

        #: np.ndarray: The image data to register to the reference.
        self.moving_data = None

        #: str: The filename of the moving data.
        self.moving_data_filename = None

        #: list[str]: A list of other filenames in the moving directory.
        self.moving_data_other_filenames = None

        #: ants.core.ants_transform.ANTsTransform: The linear affine transform.
        self.transform = None

        #: ants.core
        self.transformed_image = None

        #: tuple[slice, slice, slice]: a slicing object for the minimum bounding box.
        self.bounding_box = None

        log(self, message="Loading data...")
        channel = self.channel
        # channel = typer.prompt(
        #     text="What Channel would you like to use for the registration? \n \n "
        #          "1. CH00 \n "
        #          "2. CH01 \n "
        #          "3. CH02 \n "
        #          "4. Average \n ",
        #     default="CH00",
        #     show_default=False,
        #     show_choices=True,
        # )
        #
        # if channel == "1":
        #     channel = "CH00"
        # elif channel == "2":
        #     channel = "CH01"
        # elif channel == "3":
        #     channel = "CH02"
        # elif channel == "4":
        #     pass
        # else:
        #     log(self,
        #         message="Invalid channel. Please type 1 or CH00, 2 or CH01, etc.",
        #         level="debug"
        #     )

        self.parse_directory(
            directory=self.reference_path,
            attr_name='reference_data',
            label='Reference data',
            channel=channel)

        self.parse_directory(
            self.moving_path,
            attr_name='moving_data',
            label='Moving data',
            channel=channel)

        self.linear_registration()
        self.nonlinear_registration()
        self.export_other_images()

    def make_directories(self):
        """ Create the necessary directories for registration results. """
        for path in [
            self.saving_path,
            self.linear_path,
            self.nonlinear_path,
            self.fixed_path
        ]:
            os.makedirs(path, exist_ok=True)

    def parse_directory(self, directory, attr_name, label, channel):
        """ Parse a directory for .tif files and load the specified channel.

        Parameters
        ----------
        directory : str
            The directory to search for .tif files.
        attr_name : str
            The attribute name to set for the loaded image data.
        label : str
            A label for logging purposes, indicating the type of data being loaded.
        channel : str
            The channel to look for in the .tif files (e.g., 'CH00', 'CH01', 'CH02').

        Raises
        ------
        typer.Abort
            If the specified channel is not found in the directory, an error message
            is logged and the program is aborted.

        """
        all_tif_files = [file for file in os.listdir(directory) if
                         file.endswith('.tif')]
        target_file = None

        for file in all_tif_files:
            if channel in file:
                target_file = file
                image_path = os.path.join(directory, file)

                # Set the filename attribute
                setattr(self, attr_name + "_filename", file)

                # Load and set the image data
                setattr(self, attr_name, tifffile.imread(image_path))
                log(self, message=f"{label} loaded: {image_path}")
                break

        # Create list of other .tif files excluding the target file
        if target_file:
            other_files = [file for file in all_tif_files if
                           file != target_file]

            # Store the list of other .tif files
            setattr(self, attr_name + "_other_filenames", other_files)
            log(self, message=
                f"Found {len(other_files)} additional .tif files in {label} directory")

        if getattr(self, attr_name) is None:
            log(self, message=f"{label} not found.", level="error")

    def linear_registration(self):
        """ Perform the linear TRSAA-type registration."""
        log(self, message=f"Shape of the fixed data: {self.reference_data.shape}")
        log(self, message=f"Shape of the moving data: {self.moving_data.shape}")
        log(self, message="Beginning linear registration.")

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.transformed_image, self.transform = linear.register_image(
                moving_image=self.moving_data,
                fixed_image=self.reference_data,
                registration_type="TRSAA",
                accuracy=self.accuracy,
                verbose=True)
            log(self, message=buffer.getvalue())
            buffer.flush()

        log(self, message=f"Inspecting the linear affine transform.")
        linear.inspect_affine_transform(self.transform)

        log(self, message=f"Exporting the affine transform to: {self.linear_path}")
        clearex.registration.common.export_affine_transform(
            affine_transform=self.transform,
            directory=self.linear_path)

        log(self, message="Identifying the smallest ROI that encapsulates the data.")
        self.reference_data, self.transformed_image, self.bounding_box = crop_overlapping_datasets(
            self.reference_data,
            self.transformed_image,
            robust=True,
            lower_pct=2,
            upper_pct=98
        )
        log(self, message=f"Cropped fixed image shape: {self.reference_data.shape}")
        log(self, message=f"Cropped moving image shape: {self.transformed_image.shape}")

        log(self, message="Exporting cropped reference and moving images.")
        clearex.registration.common.export_tiff(
            image=self.reference_data,
            data_path=os.path.join(self.fixed_path, self.reference_data_filename))
        clearex.registration.common.export_tiff(
            image=self.transformed_image,
            data_path=os.path.join(self.linear_path, self.moving_data_filename))

        # If other images exist, import, transform, crop, and export.
        if hasattr(self, 'moving_data_other_filenames'):
            for file in self.moving_data_other_filenames:
                if file in self.reference_data_other_filenames:
                    log(self, message=f"Processing additional file: {file}")
                    fixed_image = tifffile.imread(
                        os.path.join(self.reference_path, file))
                    moving_image = tifffile.imread(
                        os.path.join(self.moving_path, file))

                    # Apply the linear transform
                    transformed_image = clearex.registration.linear.transform_image(
                        moving_image=moving_image,
                        fixed_image=fixed_image,
                        affine_transform=self.transform)

                    # Crop the images
                    transformed_image = transformed_image.numpy().astype(np.uint16)
                    transformed_image = transformed_image[self.bounding_box]
                    fixed_image = fixed_image[self.bounding_box]

                    # Export the transformed and cropped image
                    clearex.registration.common.export_tiff(
                        image=transformed_image,
                        data_path=os.path.join(self.linear_path, file))
                    clearex.registration.common.export_tiff(
                        image=fixed_image,
                        data_path=os.path.join(self.fixed_path, file))

        log(self, message="Linear registration complete.")

    def nonlinear_registration(self):
        """ Perform the diffeomorphic SyN warp transform. """
        log(self, message="Beginning nonlinear registration.")

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.transformed_image, transform_path = nonlinear.register_image(
                moving_image=self.transformed_image,
                fixed_image=self.reference_data,
                accuracy=self.accuracy,
                verbose=True)
            log(self, message=buffer.getvalue())
            buffer.flush()

        log(self, message=f"Exporting the registered data to: {self.nonlinear_path}")
        clearex.registration.common.export_tiff(
            image=self.transformed_image,
            data_path=os.path.join(self.nonlinear_path, self.moving_data_filename))

        log(self, message=f"Copying nonlinear warping transform from "
                           f"{transform_path} to {self.nonlinear_path}. ")
        shutil.copyfile(
            transform_path,
            os.path.join(self.nonlinear_path, 'movingToFixed1Warp.nii.gz')
        )

        log(self, message="Nonlinear registration complete.")


    def export_other_images(self):
        # If other images exist, import previously transformed and cropped data,
        # apply nonlinear transformation, and export.
        if hasattr(self, 'moving_data_other_filenames'):
            for file in self.moving_data_other_filenames:
                if file in self.reference_data_other_filenames:
                    log(self, message=f"Processing additional file: {file}")
                    fixed_image = tifffile.imread(
                        os.path.join(self.fixed_path, file))
                    moving_image = tifffile.imread(
                        os.path.join(self.linear_path, file))

                    # Convert to ants Image type.
                    fixed_image = ants.from_numpy(fixed_image)
                    moving_image = ants.from_numpy(moving_image)

                    transform_list = [
                        os.path.join(self.nonlinear_path, 'movingToFixed1Warp.nii.gz')]

                    warped_image = ants.apply_transforms(
                        fixed=fixed_image,
                        moving=moving_image,
                        transformlist=transform_list,
                        interpolator='linear'
                    )

                    # Histogram match to original data.
                    warped_image = ants.histogram_match_image(
                        source_image=warped_image,
                        reference_image=moving_image
                        # number_of_match_points=...
                        # use_threshold_at_mean_intensity=...
                    )

                    warped_image = warped_image.numpy().astype(np.uint16)

                    # Export the transformed and cropped image
                    clearex.registration.common.export_tiff(
                        image=warped_image,
                        data_path=os.path.join(self.nonlinear_path, file))

                    print("DONE?!?!?")


if __name__ == "__main__":
    Registration()