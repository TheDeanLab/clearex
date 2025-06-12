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

# Third Party Imports
import typer
import tifffile
import numpy as np

# Local Imports
import clearex.registration.linear


class Registration:
    """ A class to register data. """

    def __init__(self):
        #: str: The path to the reference data.
        self.reference_path = typer.prompt("What directory is the reference data in?")

        #: str: The path to the moving data.
        self.moving_path = typer.prompt("What directory is the moving data in?")

        # str: The path to save the data to
        self.saving_path = os.path.join(self.moving_path, "registration")
        os.makedirs(self.saving_path, exist_ok=True)

        #: str: The channel to use for registration.
        self.channel = None

        #: np.ndarray: The reference image data.
        self.reference_data = None

        #: np.ndarray: The image data to register to the reference.
        self.moving_data = None

        self.load_data()

        typer.echo(f"Starting linear registration.")

        self.linear_registration()

    def load_data(self):
        """Load the image data as indicated by the prompt."""

        def _load_data(channel):
            reference_contents = os.listdir(self.reference_path)
            for file in reference_contents:
                if channel in file:
                    image_path = str(os.path.join(self.reference_path, channel))
                    self.reference_data = tifffile.imread(image_path)
                    continue

            moving_contents = os.listdir(self.moving_path)
            for file in moving_contents:
                if channel in file:
                    image_path = str(os.path.join(self.moving_path, channel))
                    self.moving_data = tifffile.imread(image_path)
                    continue

            if self.reference_data is None:
                typer.Abort("Reference data not found.")
            if self.moving_data is None:
                typer.Abort("Moving data not found.")

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
            typer.echo("Invalid channel. Please type 1 or CH00, 2 or CH01, etc.")
            self.load_data()

    def linear_registration(self):
        """ Perform the linear TRSAA-type registration."""

        transformed_image, transform = linear.register_image(
            moving_image=self.moving_data,
            fixed_image=self.reference_data,
            registration_type="TRSAA",
            accuracy="high",
            verbose=True)

        print("Inspecting transform:")
        linear.inspect_affine_transform(transform)

        print("Exporting the affine transform to:", self.saving_path)
        linear.export_affine_transform(transform, self.saving_path)

        print("Exporting the registered data to:", self.saving_path)
        linear.export_tiff(
            image=transformed_image,
            data_path=os.path.join(self.saving_path, "registered.tif"))

        print("Done.")
