import os

from onnx.numpy_helper import to_array
from onnx.helper import make_tensor


def save_to_disk(onnx_protobuf, filename):
    dirname = os.path.dirname(filename)

    for node in onnx_protobuf.graph.node:
        for attribute in node.attribute:
            if attribute.HasField('t') and attribute.t.external_data:
                old_tensor = attribute.t
                new_tensor = make_tensor(old_tensor.name, old_tensor.data_type,
                                         old_tensor.dims, to_array(old_tensor),
                                         raw=True, external_data_dir=dirname)

                attribute.t.CopyFrom(new_tensor)

            elif len(attribute.tensors):
                for i in range(len(attribute.tensors)):
                    old_tensor = attribute.tensors[i]
                    if not old_tensor.external_data:
                        continue
                    new_tensor = make_tensor(old_tensor.name, old_tensor.data_type,
                                             old_tensor.dims, to_array(old_tensor),
                                             raw=True, external_data_dir=dirname)
                    attribute.tensors[i] = new_tensor

    with open(filename, 'wb') as f:
        f.write(onnx_protobuf.SerializeToString())


def save_to_disk_with_external_tensors(onnx_protobuf, filename):
    dirname = os.path.dirname(filename)

    for node in onnx_protobuf.graph.node:
        for attribute in node.attribute:
            if attribute.HasField('t'):
                old_tensor = attribute.t
                new_tensor = make_tensor(old_tensor.name, old_tensor.data_type,
                                         old_tensor.dims, to_array(old_tensor),
                                         raw=True, external_data_dir=dirname)

                attribute.t.CopyFrom(new_tensor)

            elif len(attribute.tensors):
                for i in range(len(attribute.tensors)):
                    old_tensor = attribute.tensors[i]
                    new_tensor = make_tensor(old_tensor.name, old_tensor.data_type,
                                             old_tensor.dims, to_array(old_tensor),
                                             raw=True, external_data_dir=dirname)
                    attribute.tensors[i] = new_tensor

    with open(filename, 'wb') as f:
        f.write(onnx_protobuf.SerializeToString())


def save_to_disk_with_internal_tensors(onnx_protobuf, filename):

    all_attributes = [attribute for node in onnx_protobuf.graph.node
                      for attribute in node.attribute]

    for attribute in all_attributes:
        if attribute.HasField('t'):
            old_tensor = attribute.t
            new_tensor = make_tensor(old_tensor.name, old_tensor.data_type,
                                     old_tensor.dims, to_array(old_tensor).tobytes(),
                                     raw=True, external_data_dir=None)

            attribute.t.CopyFrom(new_tensor)

        elif len(attribute.tensors):
            for i in range(len(attribute.tensors)):
                old_tensor = attribute.tensors[i]
                new_tensor = make_tensor(old_tensor.name, old_tensor.data_type,
                                         old_tensor.dims, to_array(old_tensor).tobytes(),
                                         raw=True, external_data_dir=None)
                attribute.tensors[i] = new_tensor

    with open(filename, 'wb') as f:
        f.write(onnx_protobuf.SerializeToString())




# def save_to_disk(onnx_protobuf, filename):
#     """Save binary protobuf with an ONNX model to a file.
#
#     If model contains tensors with `external_data` fields, those will be saved
#     in files in the same output directory.
#
#     :param filename:
#     :return:
#     """
#     with open(filename, 'wb') as f:
#         f.write(onnx_protobuf.SerializeToString())
#

# def read_from_external(protobuf_model, dir_name, read_lazy=True):
#     # store external file path as a custom field of each Tensor with external_data field...
#
#     # load model, set raw_data field to data from external files
#     # return protobuf model
#     # if read_lazy - change external_data to an absolute path
#
#
# def save_as_external_tensors(protobuf_model, dir_name):
#     # take raw_data fields, save to external files
#     # change absolute file paths to file:// URIs relative
