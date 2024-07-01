#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import enum
import inspect
import os
import sys
import time

from onnxslim.onnx_graphsurgeon.util.exception import OnnxGraphSurgeonException


# Context manager to apply indentation to messages
class LoggerIndent(object):
    def __init__(self, logger, indent):
        """Initialize LoggerIndent with a logger instance and a specified indentation level for log messages."""
        self.logger = logger
        self.old_indent = self.logger.logging_indent
        self.indent = indent

    def __enter__(self):
        """Sets the logger's indentation level when entering the context."""
        self.logger.logging_indent = self.indent
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Resets logger indentation level upon exiting the context."""
        self.logger.logging_indent = self.old_indent


# Context manager to suppress messages
class LoggerSuppress(object):
    def __init__(self, logger, severity):
        """Suppress logger messages below a specified severity level for the context duration."""
        self.logger = logger
        self.old_severity = self.logger.severity
        self.severity = severity

    def __enter__(self):
        """Set logger severity to the specified level on entering the context."""
        self.logger.severity = self.severity
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Revert the logger severity to its original level when exiting the context."""
        self.logger.severity = self.old_severity


class LogMode(enum.IntEnum):
    EACH = 0  # Log the message each time
    ONCE = 1  # Log the message only once. The same message will not be logged again.


class Logger(object):
    ULTRA_VERBOSE = -10
    VERBOSE = 0
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    SEVERITY_LETTER_MAPPING = {
        ULTRA_VERBOSE: "[UV]",
        VERBOSE: "[V]",
        DEBUG: "[D]",
        INFO: "[I]",
        WARNING: "[W]",
        ERROR: "[E]",
        CRITICAL: "[C]",
    }

    SEVERITY_COLOR_MAPPING = {
        ULTRA_VERBOSE: "cyan",
        VERBOSE: "dark_gray",
        DEBUG: "light_gray",
        INFO: "light_green",
        WARNING: "light_yellow",
        ERROR: "red_1",
        CRITICAL: "red_1",
    }

    def __init__(self, severity=INFO, colors=True, letter=True, timestamp=False, line_info=False):
        """Initialize the Logger with configurable severity, colors, letter, timestamp, and line info options."""
        self._severity = severity
        self.logging_indent = 0
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        self.once_logged = set()
        self.colors = colors
        self.letter = letter
        self.timestamp = timestamp
        self.line_info = line_info
        self.logger_callbacks = []

    @property
    def severity(self):
        """Returns the current logging severity level."""
        return self._severity

    @severity.setter
    def severity(self, value):
        """Get or set the current logging severity level."""
        self._severity = value
        for callback in self.logger_callbacks:
            callback(self._severity)

    def register_callback(self, callback):
        """Registers a callback to be invoked when the logging severity is modified."""
        callback(self._severity)
        self.logger_callbacks.append(callback)

    def indent(self, level=1):
        """Returns a context manager to indent log messages by the specified level."""
        return LoggerIndent(self, level + self.logging_indent)

    def suppress(self, severity=CRITICAL):
        """Temporarily changes logger severity, suppressing messages below the given severity level."""
        return LoggerSuppress(self, severity)

    # If once is True, the logger will only log this message a single time. Useful in loops.
    # message may be a callable which returns a message. This way, only if the message needs to be logged is it ever generated.
    def log(self, message, severity, mode=LogMode.EACH, stack_depth=2):
        """Logs a message with a specific severity and mode, supporting conditional repeated logging."""

        def process_message(message, stack_depth):
            """Generates a log message prefix with file name and line number based on the specified stack depth."""

            def get_prefix():
                def get_line_info():
                    module = inspect.getmodule(sys._getframe(stack_depth + 3)) or inspect.getmodule(
                        sys._getframe(stack_depth + 2)
                    )
                    filename = module.__file__
                    filename = os.path.relpath(filename, self.root_dir)
                    # If the file is not located in trt_smeagol, use its basename instead.
                    if os.pardir in filename:
                        filename = os.path.basename(filename)
                    return "[{:}:{:}] ".format(filename, sys._getframe(stack_depth).f_lineno)

                prefix = ""
                if self.letter:
                    prefix += f"{Logger.SEVERITY_LETTER_MAPPING[severity]} "
                if self.timestamp:
                    prefix += "({:}) ".format(time.strftime("%X"))
                if self.line_info:
                    prefix += get_line_info()
                return prefix

            def apply_indentation(message):
                """Indent each line in the message by the specified logging_indent level."""
                message_lines = str(message).splitlines()
                return "\n".join(["\t" * self.logging_indent + line for line in message_lines])

            def apply_color(message):
                """Apply color formatting to the message if color support is enabled."""
                if self.colors:
                    try:
                        import colored

                        color = Logger.SEVERITY_COLOR_MAPPING[severity]
                        return colored.stylize(message, [colored.fg(color)])
                    except ImportError:
                        self.colors = False
                        self.warning(
                            "colored module is not installed, will not use colors when logging. To enable colors, please install the colored module: python3 -m pip install colored"
                        )
                        self.colors = True
                return message

            prefix = get_prefix()
            message = apply_indentation(message)
            return apply_color("{:}{:}".format(prefix, message))

        def should_log(message):
            """Determines if a message should be logged based on the severity level and logging mode."""
            should = severity >= self._severity
            if mode == LogMode.ONCE:
                message_hash = hash(message)
                should &= message_hash not in self.once_logged
                self.once_logged.add(message_hash)
            return should

        if not should_log(message):
            return

        if callable(message):
            message = message()
        message = str(message)
        print(process_message(message, stack_depth=stack_depth))

    def ultra_verbose(self, message, mode=LogMode.EACH):
        """Logs an ultra-verbose message with a specified mode and enhanced stack depth."""
        self.log(message, Logger.ULTRA_VERBOSE, mode=mode, stack_depth=3)

    def verbose(self, message, mode=LogMode.EACH):
        """Logs a verbose message with an optional logging mode and stack depth of 3."""
        self.log(message, Logger.VERBOSE, mode=mode, stack_depth=3)

    def debug(self, message, mode=LogMode.EACH):
        """Logs a debug message with the specified mode and a stack depth of 3."""
        self.log(message, Logger.DEBUG, mode=mode, stack_depth=3)

    def info(self, message, mode=LogMode.EACH):
        """Logs an informational message with a specified mode and stack depth of 3."""
        self.log(message, Logger.INFO, mode=mode, stack_depth=3)

    def warning(self, message, mode=LogMode.EACH):
        """Logs a warning message with specified mode and stack depth of 3."""
        self.log(message, Logger.WARNING, mode=mode, stack_depth=3)

    def error(self, message, mode=LogMode.EACH):
        """Logs an error message with a specified mode and stack depth of 3."""
        self.log(message, Logger.ERROR, mode=mode, stack_depth=3)

    # Like error, but immediately exits.
    def critical(self, message):
        """Logs a critical message then raises an OnnxGraphSurgeonException with detailed context."""
        self.log(message, Logger.CRITICAL, stack_depth=3)
        raise OnnxGraphSurgeonException(message) from None  # Erase exception chain


global G_LOGGER
G_LOGGER = Logger()
