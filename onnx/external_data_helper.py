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


def write_eternal_data(value, external_data_dir, name=None):
    """Save raw bytes value to an external file."""
    name = name if name else str(uuid.uuid4())
    external_data_filename = 'data_{}.bin'.format(name)
    external_data_filepath = os.path.join(external_data_dir, external_data_filename)
    external_data_url = 'file:///{}'.format(external_data_filename)

    with open(external_data_filepath, 'wb') as data_file:
        data_file.write(value)

    return external_data_url


def read_eternal_data(external_data_url, external_data_dir):
    """Read raw bytes value from an external file."""
    url_data = urlparse(external_data_url)
    if url_data.scheme != 'file':
        raise ValueError(
            "Only file:/// URLs are supported by external_data field.")

    if not external_data_dir:
        raise RuntimeError('Please specify `external_data_dir` path to read external files.')

    external_data_filepath = os.path.join(external_data_dir, url_data.path.lstrip('/'))

    with open(external_data_filepath, 'rb') as data_file:
        value = data_file.read()

    return value
