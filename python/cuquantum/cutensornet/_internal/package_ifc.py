# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
An abstract interface to certain package-provided operations.
"""

__all__ = ['Package']

from abc import ABC, abstractmethod


class Package(ABC):

    @staticmethod
    @abstractmethod
    def get_current_stream(device_id):
        """
        Obtain the current stream on the device.

        Args:
            device_id: The id (ordinal) of the device.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def to_stream_pointer(stream):
        """
        Obtain the stream pointer.

        Args:
            stream: The stream object.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def to_stream_context(stream):
        """
        Create a context manager from the stream.

        Args:
            stream: The stream object.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_external_stream(device_id, stream_ptr):
        """
        Wrap a stream pointer into an external stream object.

        Args:
            device_id: The id (ordinal) of the device.
            stream: The stream pointer (int) to be wrapped.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_stream(device_id):
        """
        Create a new stream on the specified device.

        Args:
            device_id: The id (ordinal) of the device.
        """
        raise NotImplementedError
