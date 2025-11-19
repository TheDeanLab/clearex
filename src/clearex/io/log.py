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
import sys
import tempfile
from datetime import datetime


def initialize_logging(log_directory: str, enable_logging: bool) -> logging.Logger:
    """Initialize logging if not already configured.

    This function checks if the root logger has any handlers configured. If not,
    it initializes logging to write to a log file in the specified directory. If
    logging is already configured, it simply returns the existing root logger.

    Parameters
    ----------
    log_directory : str
        The directory path where the log file should be created if logging is not
        already configured.
    enable_logging : bool
        Flag indicating whether to initialize logging if it is not already set up.
    Returns
    -------
    logging.Logger
        The root logger instance.
    """
    root_logger = logging.getLogger()

    # Check if logger has handlers (indicates it's been configured)
    if root_logger.hasHandlers() and root_logger.level != logging.NOTSET:
        return root_logger

    # If logging is not initialized and we want to initialize it
    if enable_logging:
        return initiate_logger(log_directory=log_directory)
    else:
        # If logging is not initialized and we do not want to initialize it,
        # set a NullHandler to avoid "No handler found" warnings.
        root_logger.addHandler(logging.NullHandler())
        return root_logger


def initiate_logger(log_directory) -> logging.Logger:
    """Set up logging to a file in the specified directory.

    This function configures the root logger to write log messages to a file
    located in the specified base path. The log file is named with a timestamp
    to ensure uniqueness. The logging level is set to INFO, and the log format
    includes timestamps.

    Parameters
    ----------
    log_directory : str
        The directory where the log file will be created.
    Returns
    -------
    logging.Logger
        The configured root logger instance.
    """
    # Format: YYYY-MM-DD-HH-SS.log
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%S")
    log_filename = f"{timestamp}.log"
    log_path = os.path.join(log_directory, log_filename)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger().setLevel(logging.INFO)
    return logging.getLogger()


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


def capture_c_level_output(func, *args, **kwargs):
    """Capture stdout/stderr including C-level output from function calls."""

    # Create temporary files for stdout and stderr
    with (
        tempfile.TemporaryFile(mode="w+") as stdout_file,
        tempfile.TemporaryFile(mode="w+") as stderr_file,
    ):
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
