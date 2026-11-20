import argparse
from csv import DictReader
from pathlib import Path

import h3


def read_regions(path):
    with path.open(encoding="utf-8-sig") as file:
        return {row["h3"].strip().lower() for row in DictReader(file)}


def map_streetview(metadata, regions, output):
    resolution = h3.get_resolution(next(iter(regions)))
    with metadata.open(encoding="utf-8-sig") as file:
        for index, row in enumerate(DictReader(file)):
            cell = (row.get("h3_id") or "").strip().lower()
            if not cell:
                cell = h3.latlng_to_cell(float(row["lat"]), float(row["lon"]), resolution)
            if cell in regions:
                image = (metadata.parent / row["image_path"]).resolve(strict=True)
                folder = output / cell
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / f"{index}{image.suffix.lower()}"
                if target.is_symlink():
                    if target.resolve() == image:
                        continue
                    target.unlink()
                target.symlink_to(image)


def map_satellite(raster, regions, output):
    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    output.mkdir(parents=True, exist_ok=True)
    size = 400
    with rasterio.open(raster) as source:
        for cell in sorted(regions):
            lat, lon = zip(*h3.cell_to_boundary(cell))
            dx = max((max(lon) - min(lon)) * 0.08, 1e-6)
            dy = max((max(lat) - min(lat)) * 0.08, 1e-6)
            transform = from_bounds(min(lon) - dx, min(lat) - dy,
                                    max(lon) + dx, max(lat) + dy, size, size)
            rgb = np.zeros((3, size, size), dtype=np.uint8)
            for index in range(3):
                reproject(rasterio.band(source, index + 1), rgb[index],
                          dst_transform=transform, dst_crs="EPSG:4326",
                          resampling=Resampling.bilinear)
            ring = list(zip(lon, lat))
            polygon = {"type": "Polygon", "coordinates": [ring + [ring[0]]]}
            alpha = rasterize([(polygon, 255)], out_shape=(size, size),
                              transform=transform, dtype="uint8")
            image = Image.fromarray(rgb.transpose(1, 2, 0))
            image.putalpha(Image.fromarray(alpha))
            image.save(output / f"{cell}.png")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    for name, mapper in (("streetview", map_streetview), ("satellite", map_satellite)):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("source", type=Path)
        subparser.add_argument("regions", type=Path)
        subparser.add_argument("output", type=Path)
        subparser.set_defaults(mapper=mapper)
    args = parser.parse_args()
    args.mapper(args.source, read_regions(args.regions), args.output)


if __name__ == "__main__":
    main()
