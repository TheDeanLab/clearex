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
import typer
import os
import sys
import tempfile

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

    # Remove any NullHandlers that may have been added by other modules
    for handler in list(logger.handlers):
        if isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)

    # Prevent duplicate handlers if logger already exists
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def log_and_echo(self, message: str, level: str = "info"):
    """
    Log a message using the specified logging level and print it to the console.

    This method logs the provided message to the class logger using the given
    severity level (e.g., 'info', 'warning', 'error', 'debug'), and then echoes
    the same message to the console using `typer.echo`.

    Parameters
    ----------
    self : ...
        The class object which has the logger attribute.
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
    if level == "info":
        self.logger.info(message)
    elif level == "warning":
        self.logger.warning(message)
    elif level == "error":
        self.logger.error(message)
    elif level == "debug":
        self.logger.debug(message)
    else:
        self.logger.log(logging.INFO, message)  # fallback
    typer.echo(message)

def capture_c_level_output(func, *args, **kwargs):
    """Capture stdout/stderr including C-level output from function calls."""

    # Create temporary files for stdout and stderr
    with tempfile.TemporaryFile(mode='w+') as stdout_file, \
         tempfile.TemporaryFile(mode='w+') as stderr_file:

        # Save original file descriptors
        saved_stdout_fd = os.dup(sys.stdout.fileno())
        saved_stderr_fd = os.dup(sys.stderr.fileno())

        # Flush to avoid duplicate output
        sys.stdout.flush()
        sys.stderr.flush()

        try:
            # Redirect stdout and stderr to our temporary files
            os.dup2(stdout_file.fileno(), sys.stdout.fileno())
            os.dup2(stderr_file.fileno(), sys.stderr.fileno())

            # Call the function
            result = func(*args, **kwargs)

            # Ensure all output is written
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            # Restore original stdout and stderr
            os.dup2(saved_stdout_fd, sys.stdout.fileno())
            os.dup2(saved_stderr_fd, sys.stderr.fileno())

            # Close saved file descriptors
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)

        # Collect output
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout_content = stdout_file.read()
        stderr_content = stderr_file.read()

    return result, stdout_content, stderr_content