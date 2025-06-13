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

import imageio
# Third Party Imports
import typer
import tifffile

# Local Imports
import clearex.registration.linear as linear
import clearex.registration.nonlinear as nonlinear
from clearex import setup_logger
from clearex.file_operations.tools import crop_overlapping_datasets

class Registration:
    """ A class to register data. """

    def __init__(self):
        #: str: The path to the reference data.
        # self.reference_path = typer.prompt("What directory is the reference data in?")
        self.reference_path = ("/archive/bioinformatics/Danuser_lab/Dean/dean/2025-06"
                               "-12-registration/fixed")

        #: str: The path to the moving data.
        #self.moving_path = typer.prompt("What directory is the moving data in?")
        self.moving_path = ("/archive/bioinformatics/Danuser_lab/Dean/dean/2025-06"
                               "-12-registration/moving")

        # str: The base path to save the data to
        self.saving_path = os.path.join(self.moving_path, "registration")
        os.makedirs(self.saving_path, exist_ok=True)

        # Create the saving directory for linear registration results.
        #: str: The path to save the linear registration results.
        self.linear_path = os.path.join(self.saving_path, "linear")
        os.makedirs(self.linear_path, exist_ok=True)

        # Create the saving directory for nonlinear registration results.
        #: str: The path to save the nonlinear registration results.
        self.nonlinear_path = os.path.join(self.saving_path, "nonlinear")
        os.makedirs(self.nonlinear_path, exist_ok=True)

        # Create the fixed directory for the reference data.
        #: str: The path to save the fixed reference data.
        self.fixed_path = os.path.join(self.saving_path, "fixed")
        os.makedirs(self.fixed_path, exist_ok=True)

        #: logging.Logger: The logger for documenting registration steps.
        self.logger = setup_logger(
            name='registration',
            log_file=os.path.join(
                self.moving_path,
                "registration",
                "registration.log"),
            level=logging.DEBUG)

        # Log paths.
        self.logger.info("...Initializing new registration runtime...")
        self.logger.info(f"Moving path: {self.moving_path}")
        self.logger.info(f"Registration path: {self.reference_path}")

        #: str: The channel to use for registration.
        self.channel = "CH01"
        # self.channel = None

        #: np.ndarray: The reference image data.
        self.reference_data = None

        #: str: The filename of the reference data.
        self.reference_data_name = None

        #: np.ndarray: The image data to register to the reference.
        self.moving_data = None

        #: str: The filename of the moving data.
        self.moving_data_name = None

        #: ants.core.ants_transform.ANTsTransform: The linear affine transform.
        self.transform = None

        #: ants.core
        self.transformed_image = None

        #: tuple[slice, slice, slice]: a slicing object for the minimum bounding box.
        self.bounding_box = None

        self.load_data()
        self.linear_registration()
        self.nonlinear_registration()

    def _log_and_echo(self, message: str, level: str = "info"):
        """
        Log a message using the specified logging level and print it to the console.

        This method logs the provided message to the class logger using the given
        severity level (e.g., 'info', 'warning', 'error', 'debug'), and then echoes
        the same message to the console using `typer.echo`.

        Parameters
        ----------
        message : str
            The message to log and display.
        level : str, optional
            The severity level for logging. Must be one of:
            {'info', 'warning', 'error', 'debug'}. Defaults to 'info'.
            If an unrecognized level is provided, it falls back to 'info'.

        Returns
        -------
        None
        """
        if level == "info": self.logger.info(message)
        elif level == "warning": self.logger.warning(message)
        elif level == "error": self.logger.error(message)
        elif level == "debug": self.logger.debug(message)
        else: self.logger.log(logging.INFO, message)  # fallback
        typer.echo(message)

    def load_data(self):
        """Load the image data as indicated by the prompt."""

        self._log_and_echo("Loading data...")

        def _load_data(channel):
            def load_image_from_directory(directory, attr_name, label):
                print(os.listdir())
                for file in os.listdir(directory):
                    if channel in file:
                        image_path = os.path.join(directory, file)

                        # self.reference_data_name = "1_CH00_000.tif", etc.
                        setattr(self, attr_name + "_filename", file)

                        # self.reference_data = 3D tiff image
                        setattr(self, attr_name, tifffile.imread(image_path))
                        self._log_and_echo(f"{label} loaded: {image_path}")
                        break

                if getattr(self, attr_name) is None:
                    self._log_and_echo(f"{label} not found.", level="error")
                    raise typer.Abort

            load_image_from_directory(self.reference_path,
                                      attr_name='reference_data',
                                      label='Reference data'
                                      )
            load_image_from_directory(self.moving_path,
                                      attr_name='moving_data',
                                      label='Moving data'
                                      )

        self.channel = typer.prompt(
            text="What Channel would you like to use for the registration? \n \n "
                "1. CH00 \n "
                "2. CH01 \n "
                "3. CH02 \n "
                "4. Average \n ",
            default="CH00",
            show_default=False,
            show_choices=True,
        )

        self.channel = self.channel.lower()

        if self.channel == "ch00" or self.channel == "1": _load_data("CH00")
        elif self.channel == "ch01" or self.channel == "2": _load_data("CH01")
        elif self.channel == "ch02" or self.channel == "3": _load_data("CH02")
        elif self.channel == "average" or self.channel == "4": pass
        else:
            self._log_and_echo(
                message="Invalid channel. Please type 1 or CH00, 2 or CH01, etc.",
                level="debug"
            )
            self.load_data()

    def linear_registration(self):
        """ Perform the linear TRSAA-type registration."""
        self._log_and_echo(f"Shape of the fixed data: {self.reference_data.shape}")
        self._log_and_echo(f"ape of the moving data: {self.moving_data.shape}")
        self._log_and_echo("Beginning linear registration.")

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.transformed_image, self.transform = linear.register_image(
                moving_image=self.moving_data,
                fixed_image=self.reference_data,
                registration_type="TRSAA",
                accuracy="high",
                verbose=True)
            self.logger.info(buffer.getvalue())
            buffer.flush()

        self._log_and_echo(f"Inspecting the linear affine transform.")
        linear.inspect_affine_transform(self.transform)

        self._log_and_echo(f"Exporting the affine transform to: {linear_path}")
        linear.export_affine_transform(
            affine_transform=self.transform,
            directory=linear_path)

        self._log_and_echo("Identifying the smallest ROI that encapsulates the data.")
        self.reference_data, self.transformed_image, self.bounding_box = crop_overlapping_datasets(
            self.reference_data,
            self.transformed_image,
            robust=True,
            lower_pct=2,
            upper_pct=98
        )
        self._log_and_echo(f"Cropped fixed image shape: {self.reference_data.shape}")
        self._log_and_echo(f"Cropped moving image shape: {self.transformed_image.shape}")

        self._log_and_echo("Exporting cropped reference and moving images.")
        linear.export_tiff(
            image=self.reference_data,
            data_path=os.path.join(self.fixed_path, self.reference_data_name))
        linear.export_tiff(
            image=self.transformed_image,
            data_path=os.path.join(self.linear_path, self.moving_data_name))

        self._log_and_echo("Linear registration complete.")

    def nonlinear_registration(self):
        """ Perform the diffeomorphic SyN warp transform. """
        self._log_and_echo("Beginning nonlinear registration.")

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            transformed_image, transform_path = nonlinear.register_image(
                moving_image=self.transformed_image,
                fixed_image=self.reference_data,
                accuracy="high",
                verbose=True)
            self.logger.info(buffer.getvalue())
            buffer.flush()

        self._log_and_echo("Exporting the registered data to:", self.nonlinear_path)
        linear.export_tiff(
            image=transformed_image,
            data_path=os.path.join(self.nonlinear_path, self.moving_data_name))

        self._log_and_echo(f"Exporting nonlinear warping transform from "
                           f"{transform_path} to {self.nonlinear_path}. ")
        shutil.copyfile(
            transform_path,
            os.path.join(self.nonlinear_path, 'movingToFixed1Warp.nii.gz')
        )
        print("Nonlinear registration complete.")

if __name__ == "__main__":
    Registration()