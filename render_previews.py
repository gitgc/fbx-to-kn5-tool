#!/usr/bin/env python3
"""Render skin preview images the way Content Manager's showroom does.

    uv run render_previews.py path/to/mycar

Renders every skin in <car>/skins to <car>/skins/<skin>/preview.jpg, using
Content Manager's default preview settings (dark studio, 1022x575, 4x SSAA).
"""
import argparse
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preview_gl  # noqa: E402
import preview_scene  # noqa: E402

JPEG_QUALITY = 98


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cars", nargs="+", help="car folders")
    ap.add_argument("--skin", action="append", help="only these skins (repeatable)")
    ap.add_argument("--out", help="write here instead of <skin>/preview.jpg (single skin only)")
    args = ap.parse_args()

    renderer = preview_gl.Renderer()
    for car_dir in args.cars:
        skins = args.skin or sorted(d for d in os.listdir(os.path.join(car_dir, "skins"))
                                    if os.path.isdir(os.path.join(car_dir, "skins", d)))
        car = preview_scene.Car(car_dir)
        camera = preview_scene.preview_camera(car, renderer.width, renderer.height)
        for skin in skins:
            start = time.time()
            car.set_skin(skin)
            image = renderer.render(car, camera)
            out = args.out or os.path.join(car_dir, "skins", skin, "preview.jpg")
            Image.fromarray(image).save(out, quality=JPEG_QUALITY, subsampling=0)
            print(f"{out} ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
