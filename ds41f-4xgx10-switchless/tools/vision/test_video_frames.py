"""CPU tests for video sampling; requires PyAV and Pillow, no model or GPU."""
import base64
from fractions import Fraction
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

import av
from PIL import Image
from video_frames import sample


def clip():
    buf = io.BytesIO()
    with av.open(buf, 'w', format='mp4') as output:
        stream = output.add_stream('mpeg4', rate=1)
        stream.width = stream.height = 64
        stream.pix_fmt = 'yuv420p'
        for i, color in enumerate(['red', 'green', 'blue']):
            frame = av.VideoFrame.from_image(Image.new('RGB', (64, 64), color))
            frame.pts, frame.time_base = i, Fraction(1, 1)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return buf.getvalue()


class SamplingTests(unittest.TestCase):
    def test_timestamps_and_scene_order(self):
        result = sample(clip(), 3)
        self.assertEqual(result['duration'], 3.0)
        self.assertEqual([f['timestamp'] for f in result['frames']], [0.0, 1.0, 2.0])
        for frame, channel in [(result['frames'][0], 0), (result['frames'][-1], 2)]:
            image = Image.open(io.BytesIO(base64.b64decode(frame['url'].split(',', 1)[1])))
            self.assertEqual(image.size, (64, 64))
            pixel = image.convert('RGB').getpixel((32, 32))
            self.assertGreater(pixel[channel], 200)
            self.assertTrue(all(v < 30 for i, v in enumerate(pixel) if i != channel))

    def test_frame_count_caps_and_single_frame(self):
        raw = clip()
        self.assertEqual(len(sample(raw, 100)['frames']), 3)
        one = sample(raw, 1)['frames']
        self.assertEqual(len(one), 1)
        self.assertEqual(one[0]['timestamp'], 0.0)
        for count in [0, 101]:
            with self.assertRaises(ValueError):
                sample(raw, count)

    def test_broken_clip(self):
        with self.assertRaises(Exception):
            sample(b'not a video', 3)

    def test_audio_without_video_is_rejected(self):
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(8000)
            output.writeframes(b'\x00' * 16000)
        with self.assertRaisesRegex(ValueError, 'no video stream'):
            sample(buf.getvalue(), 3)


class PathTests(unittest.TestCase):
    """The clip is a file that PyAV reads as it decodes."""

    def padded_clip(self):
        # a free box after the clip: the file is far larger than any one read
        return clip() + struct.pack('>I4s', 8 + (2 << 20), b'free') + b'\0' * (2 << 20)

    def test_a_path_is_opened_and_read_in_pieces(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'clip')
            with open(path, 'wb') as f:
                f.write(self.padded_clip())
            result = sample(path, 3)
            self.assertEqual([f['timestamp'] for f in result['frames']], [0.0, 1.0, 2.0])
            reads = []

            class Counting(io.FileIO):
                def readinto(self, b):
                    reads.append(len(b))
                    return super().readinto(b)

            with io.BufferedReader(Counting(path, 'rb'), buffer_size=1 << 15) as f:
                self.assertEqual(len(sample(f, 3)['frames']), 3)
            self.assertTrue(reads)
            self.assertLess(max(reads), 1 << 20, 'no read takes the file whole')

    def test_the_command_line_takes_a_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'clip')
            with open(path, 'wb') as f:
                f.write(clip())
            r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'video_frames.py'),
                                path, '--max-frames', '2'], capture_output=True, timeout=60)
            self.assertEqual((r.returncode, len(json.loads(r.stdout)['frames'])), (0, 2), r.stderr)

    def test_bytes_are_still_taken(self):
        self.assertEqual(len(sample(clip(), 3)['frames']), 3)


if __name__ == '__main__':
    unittest.main()
