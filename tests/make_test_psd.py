"""Writes minimal, well-formed test PSDs (raw composite, no Version Info resource, i.e. a
"non-Photoshop writer") from an 8-bit raw dump, optionally embedding an ICC profile.

  python make_test_psd.py rgb|gray 8|32 <in.raw> <width> <height> <out.psd> [profile.icc]

in.raw is interleaved 8-bit RGB (rgb) or 8-bit gray (gray), e.g. from
`magick src.png -depth 8 rgb:in.raw`. 32-bit output is linear-light float (sRGB decoded),
like Photoshop's 32-bit mode."""
import struct
import sys
from array import array

mode, depth, src, w, h, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
icc = open(sys.argv[7], 'rb').read() if len(sys.argv) > 7 else b''
ch = 3 if mode == 'rgb' else 1
data = open(src, 'rb').read()
assert len(data) == w * h * ch, 'raw size mismatch'

def to_linear(v):
    s = v / 255.0
    return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4

resources = b''
if icc:  # image resource 1039 (0x040F): ICC profile
    resources = b'8BIM' + struct.pack('>H', 0x040F) + b'\0\0' + struct.pack('>I', len(icc)) + icc + (b'\0' if len(icc) & 1 else b'')

with open(dst, 'wb') as f:
    f.write(b'8BPS' + struct.pack('>H6xHIIHH', 1, ch, h, w, depth, 3 if mode == 'rgb' else 1))
    f.write(struct.pack('>I', 0))                       # colour mode data
    f.write(struct.pack('>I', len(resources)) + resources)
    f.write(struct.pack('>I', 0))                       # layer & mask info
    f.write(struct.pack('>H', 0))                       # raw composite
    lut = [to_linear(i) for i in range(256)]
    for c in range(ch):
        plane = data[c::ch]
        if depth == 8:
            f.write(plane)
        else:
            a = array('f', (lut[v] for v in plane))
            a.byteswap()  # big-endian
            f.write(a.tobytes())
print('wrote', dst)
