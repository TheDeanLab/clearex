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
import logging
import os

# Third Party Imports
import typer

# Local Imports
from clearex.registration import Registration

app = typer.Typer(
    help="Command line interface for ClearEx",
    name="clearex",
    add_help_option=True
)

CLEAR_EX_LOGO = r"""
       _                          
      | |                         
   ___| | ___  __ _ _ __ _____  __
  / __| |/ _ \/ _` | '__/ _ \ \/ /
 | (__| |  __/ (_| | | |  __/>  < 
  \___|_|\___|\__,_|_|  \___/_/\_\
                                  
"""

def initiate_logger(base_path):
    log_path = os.path.join(base_path, "distance3.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    logging.getLogger().setLevel(logging.INFO)

def run_deconvolution():
    reference = typer.prompt("Location of the data to deconvolve")
    moving = typer.prompt("Location of the PSF")
    output = typer.prompt("Location to save the data")
    typer.echo("Starting deconvolution...")


@app.command()
def main():
    """Run the ClearEx command line interface."""
    typer.echo(CLEAR_EX_LOGO)
    operation = typer.prompt(
        text="What type of image operation would you like to perform? \n \n "
             "1. Registration \n "
             "2. Deconvolution \n \n",
        default="registration",
        show_default=False,
        show_choices=True,
    )

    if operation.lower() == "registration" or operation == "1":
        Registration()
    elif operation.lower() == "deconvolution" or operation == "2":
        run_deconvolution()
    else:
        typer.echo("Only the registration and deconvolution routines are currently "
                   "available.")
        raise typer.Exit(code=1)

typer.echo("Done.")



if __name__ == "__main__":
    typer.run(app())
