from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import uuid

try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse


def write_external_data(value, external_data_dir, name=None):
    """Save raw bytes value to an external file."""
    name = name if name else str(uuid.uuid4())
    external_data_filename = 'data_{}.bin'.format(name)
    external_data_filepath = os.path.join(external_data_dir, external_data_filename)
    external_data_url = 'file:///{}'.format(external_data_filename)

    with open(external_data_filepath, 'wb') as data_file:
        data_file.write(value)

    return external_data_url


def _data_url_to_absolute_path(external_data_url, external_data_dir):
    url_data = urlparse(external_data_url)
    if url_data.scheme != 'file':
        raise ValueError(
            "Only file:// URIs are supported by external_data field.")

    if not external_data_dir:
        raise RuntimeError('Please specify `external_data_dir` path to read external files.')

    external_data_filename = url_data.path.lstrip('/')
    external_data_filepath = os.path.join(external_data_dir, external_data_filename)
    return external_data_filepath


def read_external_data(external_data_filepath):
    """Read raw bytes value from an external file."""
    with open(external_data_filepath, 'rb') as data_file:
        value = data_file.read()

    return value


def _get_all_tensors(onnx_protobuf):
    all_tensors = []
    for node in onnx_protobuf.graph.node:
        for attribute in node.attribute:
            all_tensors.append(attribute.t)
            all_tensors.extend(attribute.tensors)
    return all_tensors


def set_runtime_external_data_path_on_tensors(onnx_protobuf, onnx_protobuf_file_path):
    external_data_dir = os.path.dirname(onnx_protobuf_file_path)
    for tensor in _get_all_tensors(onnx_protobuf):
        if tensor.HasField("external_data"):
            absolute_path = _data_url_to_absolute_path(tensor.external_data, external_data_dir)
            tensor.external_data_runtime_path = absolute_path
    return onnx_protobuf
