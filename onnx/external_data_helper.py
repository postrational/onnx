from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import uuid
from itertools import chain

from six.moves.urllib.parse import urlparse
from onnx import load_from_string, numpy_helper


def get_all_tensors(onnx_model_proto):
    """Scan an ONNX model for all tensors and return as an iterator."""
    return chain(get_initializer_tensors(onnx_model_proto),
                 get_attribute_tensors(onnx_model_proto))


def get_initializer_tensors(onnx_model_proto):
    """Create an iterator of initializer tensors from ONNX model."""
    for initializer in onnx_model_proto.graph.initializer:
        yield initializer


def get_attribute_tensors(onnx_model_proto):
    """Create an iterator of tensors from node attributes of an ONNX model."""
    for node in onnx_model_proto.graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            for tensor in attribute.tensors:
                yield tensor


def generate_persistence_value(tensor_name=None):
    """Create a persistence value for the `external_data` field."""
    name = tensor_name if tensor_name else str(uuid.uuid4())
    external_data_filename = 'data_{}.bin'.format(name)
    return 'file:///{}'.format(external_data_filename)


def persistence_to_filename(persistence_value):
    """Parse persistence value for the `external_data`, return filename.

    :param persistence_value: `file://` value of the `external_data` field
    :return: string containing the relative path to a data file
    """
    uri_data = urlparse(persistence_value)
    if uri_data.scheme != 'file':
        raise ValueError(
            "Only file:// URIs are supported by external_data field.")
    filename = uri_data.path.lstrip('/.')
    return filename


def persistence_to_runtime(persistence_value, basedir):
    """Convert persistence value of `external_data` to a runtime value.

    The runtime value contains both the external data filename and the
    directory in which the main ONNX model file was store.

    :param persistence_value: `file://` value of the `external_data` field
    :param basedir: directory containing the main ONNX model file
    :return: string with the `runtime://` value for `external_data`
    """
    data_filename = persistence_to_filename(persistence_value)
    runtime_value = 'runtime://{}#{}'.format(basedir, data_filename)
    return runtime_value


def runtime_to_persistence(runtime_value):  # -> persistence_value
    """Convert runtime value of `external_data` to a persistence value.

    :param runtime_value: `runtime://` value of the `external_data` field
    :return: string with the `file://` value for `external_data`
    """
    uri_data = urlparse(runtime_value)
    return 'file:///{}'.format(uri_data.fragment)


def set_external_data_runtime_values(onnx_model_proto, onnx_filename):
    """Convert persistence values to runtime values in an ONNX model.

    Iterate over all tensors in an ONNX model and convert the `external_data`
    field values from persistence (`file://`) form to `runtime://` form.

    :param onnx_model_proto: Loaded ONNX model
    :param onnx_filename: Path to file from which ONNX model was loaded.
    """
    basedir = os.path.dirname(onnx_filename)
    for tensor in get_all_tensors(onnx_model_proto):
        if tensor.HasField("external_data"):
            persistence_value = tensor.external_data
            runtime_value = persistence_to_runtime(persistence_value, basedir)
            tensor.external_data = runtime_value


def load_from_disk(onnx_filename):
    """Load binary protobuf file with an ONNX model.

    :param onnx_filename: Path to file containing an ONNX model.
    :return: loaded ONNX model
    """
    with open(onnx_filename, 'rb') as f:
        onnx_string = f.read()
    onnx_model_proto = load_from_string(onnx_string)

    set_external_data_runtime_values(onnx_model_proto, onnx_filename)

    return onnx_model_proto


def save_to_disk(onnx_model_proto, filename):
    """Save ONNX model to files on disk.

    External data is written to additional files relative to the directory
    in which the ONNX file is written.

    :param onnx_model_proto: ONNX Protocol Buffers model
    :param filename: path to the output file
    """
    dirname = os.path.dirname(filename)

    for tensor in get_all_tensors(onnx_model_proto):
        if tensor.HasField("external_data"):
            if tensor.external_data.startswith('runtime://'):
                persistence_val = runtime_to_persistence(tensor.external_data)
                tensor.external_data = persistence_val

            data_filename = persistence_to_filename(tensor.external_data)
            external_data_filepath = os.path.join(dirname, data_filename)

            tensor_value = numpy_helper.to_array(tensor)

            # Write external data file
            with open(external_data_filepath, 'wb') as data_file:
                data_file.write(tensor_value.tobytes())

            # Clear tensor data fields
            for data_field in ['double_data', 'float_data', 'int32_data',
                               'int64_data', 'raw_data',
                               'string_data', 'uint64_data']:
                tensor.ClearField(data_field)

    with open(filename, 'wb') as f:
        f.write(onnx_model_proto.SerializeToString())
