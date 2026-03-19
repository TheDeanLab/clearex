#  Copyright (c) 2021-2025  The University of Texas Southwestern Medical Center.
#  All rights reserved.

import logging

import clearex.io.log as log_module


def test_initiate_logger_raises_distributed_proxy_logger_to_warning(tmp_path) -> None:
    root_logger = logging.getLogger()
    previous_handlers = list(root_logger.handlers)
    previous_root_level = root_logger.level
    library_logger = logging.getLogger("distributed.http.proxy")
    previous_library_level = library_logger.level

    try:
        library_logger.setLevel(logging.NOTSET)

        logger = log_module.initiate_logger(tmp_path)

        assert logger is root_logger
        assert logging.getLogger("distributed.http.proxy").level == logging.WARNING
    finally:
        for handler in list(root_logger.handlers):
            handler.close()
            root_logger.removeHandler(handler)
        for handler in previous_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(previous_root_level)
        library_logger.setLevel(previous_library_level)
