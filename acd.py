#!/usr/bin/env python3
"""Pack and unpack Assetto Corsa data.acd files.

    python3 acd.py mycar             # pack mycar/data/ into mycar/data.acd
    python3 acd.py --unpack mycar    # unpack mycar/data.acd into mycar/data/

A data.acd holds the files of a car's data/ folder. Each entry is its name, its
length and one little-endian int32 per byte: the byte plus a character of a key
derived from the car's folder name, so the folder name must match the installed
one. Entries are written in Content Manager's order (case-insensitive by name)
and file bytes are kept as they are, so repacking an unpacked file gives back
the same file.
"""
import argparse
import os
import struct
import sys

HEADER = -1111  # newer Kunos files start with this and a version int; Content Manager writes none


def _cdiv(a, b):
    """C integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _i32(x):
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >= 1 << 31 else x


def key(folder_name):
    """The encryption key for a car folder name, e.g. "202-124-245-48-230-253-2-103"."""
    c = [ord(ch) for ch in folder_name]
    n = len(c)
    k1 = sum(c)
    k2 = 0
    for i in range(0, n - 1, 2):
        k2 = _i32(_i32(k2 * c[i]) - c[i + 1])
    k3 = 0
    for i in range(1, n - 3, 3):
        k3 = _i32(_cdiv(_i32(k3 * c[i]), c[i + 1] + 27) - 27 - c[i - 1])
    k4 = 0x1683 - sum(c[1:])
    k5 = 0x42
    for i in range(1, n - 4, 4):
        k5 = _i32((c[i] + 15) * k5 * (c[i - 1] + 15) + 22)
    k6 = 0x65 - sum(c[0:n - 2:2])
    k7 = 0xAB
    for i in range(0, n - 2, 2):
        k7 -= _cdiv(k7, c[i]) * c[i]
    k8 = 0xAB
    for i in range(0, n - 1):
        k8 = _cdiv(k8, c[i]) + c[i + 1]
    return "-".join(str(k & 0xFF) for k in (k1, k2, k3, k4, k5, k6, k7, k8))


def read(path, folder_name=None):
    """Entries of a data.acd as [(name, bytes)], in file order."""
    k = key(folder_name or _folder_name(path)).encode("ascii")
    with open(path, "rb") as f:
        b = f.read()
    p = 8 if len(b) >= 4 and struct.unpack_from("<i", b)[0] == HEADER else 0
    entries = []
    while p < len(b):
        n, = struct.unpack_from("<i", b, p)
        name = b[p + 4:p + 4 + n].decode("ascii")
        p += 4 + n
        length, = struct.unpack_from("<i", b, p)
        values = struct.unpack_from(f"<{length}i", b, p + 4)
        p += 4 + 4 * length
        entries.append((name, bytes((v - k[i % len(k)]) & 0xFF for i, v in enumerate(values))))
    return entries


def write(path, entries, folder_name=None):
    """Write [(name, bytes)] to a data.acd, in the given order."""
    k = key(folder_name or _folder_name(path)).encode("ascii")
    out = bytearray()
    for name, data in entries:
        encoded = name.encode("ascii")
        out += struct.pack("<i", len(encoded)) + encoded + struct.pack("<i", len(data))
        out += struct.pack(f"<{len(data)}i", *(c + k[i % len(k)] for i, c in enumerate(data)))
    with open(path, "wb") as f:
        f.write(out)


def _folder_name(acd_path):
    return os.path.basename(os.path.dirname(os.path.abspath(acd_path)))


def pack(car_dir):
    """Pack <car>/data/ into <car>/data.acd; returns the number of files."""
    data_dir = os.path.join(car_dir, "data")
    names = sorted((f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))
                    and not f.startswith(".")), key=str.upper)
    entries = []
    for name in names:
        with open(os.path.join(data_dir, name), "rb") as f:
            entries.append((name, f.read()))
    write(os.path.join(car_dir, "data.acd"), entries)
    return len(entries)


def unpack(car_dir):
    """Unpack <car>/data.acd into <car>/data/; returns the number of files."""
    entries = read(os.path.join(car_dir, "data.acd"))
    data_dir = os.path.join(car_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    for name, data in entries:
        with open(os.path.join(data_dir, name), "wb") as f:
            f.write(data)
    return len(entries)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cars", nargs="+", help="car folders, named as installed (the key comes from the name)")
    ap.add_argument("--unpack", action="store_true", help="unpack data.acd into data/ instead")
    args = ap.parse_args()
    for car_dir in args.cars:
        car_dir = os.path.abspath(car_dir)
        if args.unpack:
            print(f"unpacked {unpack(car_dir)} files to {os.path.join(car_dir, 'data')}")
        else:
            print(f"packed {pack(car_dir)} files into {os.path.join(car_dir, 'data.acd')}")


if __name__ == "__main__":
    sys.exit(main())
