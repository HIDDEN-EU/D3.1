#!/usr/bin/env python3
"""Stream ROS 2 MCAP recordings into NumPy, compressed images and JSON.

Install: python -m pip install mcap mcap-ros2-support numpy
Run: python mcap_pipeline.py recording.mcap --output extracted
     python mcap_pipeline.py recordings/ --output extracted --topics /velodyne_points

Requires embedded ros2msg schemas and CDR messages. Original files are untouched.
Each run uses a NEW output directory. No synchronization or coordinate transforms
are applied. Failed runs remain explicitly marked by INCOMPLETE.
"""
import argparse
import base64
import hashlib
import json
import re
from pathlib import Path
import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def pointcloud_array(msg):
    """Honor PointField offsets, counts, byte order, and row padding."""
    types = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4',
             6: 'u4', 7: 'f4', 8: 'f8'}
    names, formats, offsets = [], [], []
    for field in msg.fields:
        if field.datatype not in types or field.count < 1:
            raise ValueError('Unsupported PointField: ' + field.name)
        dtype = np.dtype(('>' if msg.is_bigendian else '<') + types[field.datatype])
        names.append(field.name)
        formats.append(dtype if field.count == 1 else (dtype, (field.count,)))
        offsets.append(field.offset)
    dtype = np.dtype(dict(names=names, formats=formats, offsets=offsets,
                         itemsize=msg.point_step))
    if msg.row_step < msg.width * msg.point_step:
        raise ValueError('Invalid PointCloud2 row_step')
    return np.ndarray((msg.height, msg.width), dtype=dtype,
                      buffer=bytes(msg.data), strides=(msg.row_step, msg.point_step))


def image_array(msg):
    """Decode images; convert bayer_rggb8 to RGB, preserve other encodings."""

    if msg.encoding.lower() == 'bayer_rggb8':
        if msg.step < msg.width:
            raise ValueError('Invalid Bayer image step')

        # One byte per pixel; respect padding at the end of each row.
        bayer = np.ndarray(
            shape=(msg.height, msg.width),
            dtype=np.uint8,
            buffer=bytes(msg.data),
            strides=(msg.step, 1),
        )

        return cv2.cvtColor(
            np.ascontiguousarray(bayer),
            cv2.COLOR_BayerBG2RGB,
        )

    """Keep original channel order, precision and endianness (no normalization)."""
    known = {'rgb8': ('u1', 3), 'bgr8': ('u1', 3),
             'rgba8': ('u1', 4), 'bgra8': ('u1', 4),
             'mono8': ('u1', 1), 'mono16': ('u2', 1),
             'rgb16': ('u2', 3), 'bgr16': ('u2', 3)}
    if msg.encoding in known:
        code, channels = known[msg.encoding]
    else:
        match = re.fullmatch(r'(8|16|32|64)([USF])C([1-9][0-9]*)', msg.encoding)
        if not match:
            raise ValueError('Unsupported image encoding: ' + msg.encoding)
        bits, kind, channels = match.groups()
        code = {'U': 'u', 'S': 'i', 'F': 'f'}[kind] + str(int(bits) // 8)
        channels = int(channels)
    dtype = np.dtype(('>' if msg.is_bigendian else '<') + code)
    pixel_step = dtype.itemsize * channels
    if msg.step < msg.width * pixel_step:
        raise ValueError('Invalid image step')
    arr = np.ndarray((msg.height, msg.width, channels), dtype=dtype,
                     buffer=bytes(msg.data),
                     strides=(msg.step, pixel_step, dtype.itemsize))
    return arr[..., 0] if channels == 1 else arr


def json_value(obj):
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return {'base64': base64.b64encode(obj).decode('ascii')}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, '__slots__'):
        return {name: getattr(obj, name) for name in obj.__slots__
                if not name.startswith('_')}
    if hasattr(obj, '__dict__'):
        return vars(obj)
    raise TypeError('Cannot serialize ' + type(obj).__name__)


def save_message(schema_name, msg, stem):
    details = {}
    if schema_name == 'sensor_msgs/msg/PointCloud2':
        path = stem.with_suffix('.npy')
        np.save(path, pointcloud_array(msg), allow_pickle=False)
        details = {'is_dense': msg.is_dense}
    elif schema_name == 'sensor_msgs/msg/Image':
        path = stem.with_suffix('.npy')
        arr = image_array(msg)
        np.save(path, arr, allow_pickle=False)

        details = {
            'encoding': (
                'rgb8'
                if msg.encoding.lower() == 'bayer_rggb8'
                else msg.encoding
            ),
            'source_encoding': msg.encoding,
            'shape': list(arr.shape),
            'dtype': str(arr.dtype),
        }
    elif schema_name == 'sensor_msgs/msg/CompressedImage':
        payload = bytes(msg.data)
        # Signature check avoids mislabeling compressedDepth payloads as PNG.
        suffix = '.jpg' if payload.startswith(b'\xff\xd8\xff') else (
            '.png' if payload.startswith(b'\x89PNG\r\n\x1a\n') else '.bin')
        path = stem.with_suffix(suffix)
        path.write_bytes(payload)
        details = {'format': msg.format}
    else:
        path = stem.with_suffix('.json')
        path.write_text(json.dumps(msg, default=json_value), encoding='utf-8')
    return path, details


def ingest(source, output, topics=None):
    files = sorted(source.rglob('*.mcap')) if source.is_dir() else [source]
    if not files or any(not p.is_file() for p in files):
        raise FileNotFoundError('No input MCAP files found')
    output.mkdir(parents=True, exist_ok=False)
    marker = output / 'INCOMPLETE'
    marker.write_text('Processing has not completed.\n')
    total = 0
    with (output / 'index.jsonl').open('w', encoding='utf-8') as index:
        for recording_number, file in enumerate(files):
            recording_id = 'recording_{:04d}'.format(recording_number)
            print('Reading', file, flush=True)
            with file.open('rb') as stream:
                reader = make_reader(stream, decoder_factories=[DecoderFactory()])
                messages = reader.iter_decoded_messages(topics=topics, log_time_order=False)
                for i, (schema, channel, message, decoded) in enumerate(messages):
                    slug = re.sub(r'[^A-Za-z0-9_-]', '_', channel.topic)[:80]
                    topic_id = slug + '_' + hashlib.sha256(channel.topic.encode()).hexdigest()[:12]
                    folder = output / recording_id / topic_id
                    folder.mkdir(parents=True, exist_ok=True)
                    path, details = save_message(schema.name, decoded,
                                                 folder / '{:09d}_{}'.format(i, message.log_time))
                    header = getattr(decoded, 'header', None)
                    stamp = getattr(header, 'stamp', None)
                    row = {'source': str(file.resolve()), 'recording_id': recording_id,
                           'topic': channel.topic, 'schema': schema.name,
                           'sequence': message.sequence,
                           'log_time_ns': message.log_time,
                           'publish_time_ns': message.publish_time,
                           'sensor_time_ns': None if stamp is None else (
                               int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)),
                           'frame_id': getattr(header, 'frame_id', None),
                           'path': path.relative_to(output).as_posix(), **details}
                    index.write(json.dumps(row) + '\n')
                    total += 1
                    if total % 1000 == 0:
                        print('Saved', total, 'messages', flush=True)
    marker.unlink()
    (output / 'COMPLETE.json').write_text(json.dumps({'messages': total, 'files': len(files)}))
    print('Done:', total, 'messages saved to', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='MCAP file or directory')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--topics', nargs='+', help='Optional topic names; default: all')
    args = parser.parse_args()
    ingest(args.source, args.output, args.topics)



# E:\dev\hidden\data\iccs_collaborative_dataset