# modules/terrain.py
from math import atan2, cos, degrees, radians, sqrt

import numpy as np
import pandas as pd
from PIL import Image


def read_tif_simple(file_or_path):
    image = Image.open(file_or_path)
    elevation = np.array(image, dtype=np.float32)
    return elevation[:, :, 0] if elevation.ndim > 2 else elevation


def load_dem_data(file_or_path):
    elevation = read_tif_simple(file_or_path)
    gy, gx = np.gradient(elevation)
    cell_size_x = 30 * cos(radians(40))
    cell_size_y = 30
    return {
        "elevation": elevation,
        "gx": gx,
        "gy": gy,
        "cell_size": (cell_size_x + cell_size_y) / 2,
        "shape": elevation.shape,
    }


def query_dem_at_point(dem_data, lon: float, lat: float) -> dict:
    if dem_data is None:
        return {"slope": 0, "aspect": 0, "elevation": 1000}
    rows, cols = dem_data["shape"]
    min_lon, max_lon = 114.69, 115.04
    max_lat, min_lat = 44.10, 44.01
    col = int((lon - min_lon) / (max_lon - min_lon) * cols)
    row = int((max_lat - lat) / (max_lat - min_lat) * rows)
    col = max(0, min(col, cols - 1))
    row = max(0, min(row, rows - 1))
    dz_dx = float(dem_data["gx"][row, col])
    dz_dy = float(dem_data["gy"][row, col])
    rise = sqrt(dz_dx ** 2 + dz_dy ** 2)
    return {
        "slope": degrees(atan2(rise, dem_data["cell_size"])),
        "aspect": (degrees(atan2(-dz_dx, dz_dy)) + 360) % 360,
        "elevation": float(dem_data["elevation"][row, col]),
    }


def load_tower_coordinates(excel_file, tower_nums=None) -> dict:
    df = pd.read_excel(excel_file)
    name_col = next(col for col in df.columns if "运行编号" in str(col) or "设备名称" in str(col))
    lon_col = next(col for col in df.columns if "经度" in str(col) or "X坐标" in str(col))
    lat_col = next(col for col in df.columns if "纬度" in str(col) or "Y坐标" in str(col))
    output = {}
    for _, row in df.iterrows():
        tower_num = int("".join(ch for ch in str(row[name_col]) if ch.isdigit())[-3:])
        if tower_nums is None or tower_num in tower_nums:
            output[tower_num] = {"lon": float(row[lon_col]), "lat": float(row[lat_col])}
    return output


def build_terrain_lookup(dem_data, tower_coords: dict, weather_positions: list) -> dict:
    if dem_data is None or not tower_coords:
        return {idx: {"slope": 0, "aspect": 0, "elevation": 1000} for idx, _ in enumerate(weather_positions)}
    output = {}
    for idx, position in enumerate(weather_positions):
        coord = tower_coords.get(position)
        output[idx] = query_dem_at_point(dem_data, coord["lon"], coord["lat"]) if coord else {
            "slope": 0,
            "aspect": 0,
            "elevation": 1000,
        }
    return output
