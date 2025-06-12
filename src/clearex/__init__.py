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

import logging
import os


def setup_logger(name='my_logger', log_file='app.log', level=logging.INFO):
    """Set up a basic logger that writes to a file and the console.

    Parameters
    ----------
    name : str
        Name of the logger.
    log_file : str
        File path for the log file.
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns
    -------
    logger : logging.Logger
        Configured logger instance.
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Create or get the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if logger already exists
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
