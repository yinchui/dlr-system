from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import atan, atan2, cos, degrees, hypot, isfinite, radians, sin
from numbers import Integral, Real
import os
import re
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from affine import Affine
from pyproj import Geod, Transformer
from pyproj.exceptions import CRSError, ProjError
import rasterio
from rasterio.io import DatasetReaderBase, MemoryFile
from rasterio.transform import rowcol, xy


_DEM_KEYS = ("elevation", "mask", "crs", "transform", "bounds", "shape", "nodata")
_SAMPLE_KEYS = ("slope", "aspect", "elevation", "source", "reason")
_WGS84_GEOD = Geod(ellps="WGS84")
_WGS84_CRS = "EPSG:4326"


@dataclass(frozen=True)
class DemGrid(Mapping[str, Any]):
    elevation: np.ndarray
    mask: np.ndarray
    crs: Any
    transform: Affine
    bounds: Any
    nodata: Any
    shape: tuple[int, int] = field(init=False)

    def __post_init__(self):
        elevation = np.array(self.elevation, copy=True)
        mask = np.array(self.mask, dtype=bool, copy=True)
        if elevation.ndim != 2:
            raise ValueError("DEM first band must be a two-dimensional array")
        if mask.shape != elevation.shape:
            raise ValueError("DEM mask shape must match elevation shape")
        elevation.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "elevation", elevation)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "shape", elevation.shape)

    def __getitem__(self, key: str) -> Any:
        if key not in _DEM_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(_DEM_KEYS)

    def __len__(self) -> int:
        return len(_DEM_KEYS)


@dataclass(frozen=True)
class TerrainSample(Mapping[str, Any]):
    slope: float
    aspect: float
    elevation: float
    source: str
    reason: str | None = None

    def __getitem__(self, key: str) -> Any:
        if key not in _SAMPLE_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(_SAMPLE_KEYS)

    def __len__(self) -> int:
        return len(_SAMPLE_KEYS)


@dataclass(frozen=True)
class TerrainLoadAttempt:
    dem_data: Any
    tower_coords: Any

    @property
    def success(self) -> bool:
        return self.dem_data is not None and bool(self.tower_coords)


class MissingTowerColumnsError(ValueError):
    pass


def _default_sample(reason: str) -> TerrainSample:
    return TerrainSample(
        slope=0.0,
        aspect=0.0,
        elevation=1000.0,
        source="default",
        reason=reason,
    )


@contextmanager
def _open_raster(file_or_path):
    if isinstance(file_or_path, DatasetReaderBase):
        yield file_or_path
        return

    if isinstance(file_or_path, MemoryFile):
        with file_or_path.open() as dataset:
            yield dataset
        return

    original_position = None
    if not isinstance(file_or_path, (str, bytes, os.PathLike)):
        try:
            original_position = file_or_path.tell()
            file_or_path.seek(0)
        except (AttributeError, OSError):
            original_position = None

    try:
        with rasterio.open(file_or_path) as dataset:
            yield dataset
    finally:
        if original_position is not None:
            try:
                file_or_path.seek(original_position)
            except (AttributeError, OSError, ValueError):
                pass


def load_dem_data(file_or_path) -> DemGrid:
    with _open_raster(file_or_path) as dataset:
        if dataset.count < 1:
            raise ValueError("DEM must contain at least one raster band")
        band = dataset.read(1, masked=True)
        elevation = np.array(band.data, copy=True)
        mask = np.ma.getmaskarray(band).astype(bool, copy=True)
        mask |= ~np.isfinite(elevation)
        return DemGrid(
            elevation=elevation,
            mask=mask,
            crs=dataset.crs,
            transform=dataset.transform,
            bounds=dataset.bounds,
            nodata=dataset.nodata,
        )


def read_tif_simple(file_or_path) -> np.ndarray:
    """Compatibility entry point backed by the canonical rasterio loader."""
    return load_dem_data(file_or_path).elevation


def _has_valid_georeference(dem: DemGrid) -> bool:
    if dem.crs is None or dem.transform is None:
        return False
    coefficients = np.asarray(tuple(dem.transform)[:6], dtype=float)
    if not np.isfinite(coefficients).all():
        return False
    determinant = dem.transform.a * dem.transform.e - dem.transform.b * dem.transform.d
    if not isfinite(determinant) or abs(determinant) <= np.finfo(float).eps:
        return False
    default_transforms = (Affine.identity(), Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0))
    return not any(dem.transform.almost_equals(value) for value in default_transforms)


def _to_dem_transformer(dem: DemGrid) -> Transformer:
    return Transformer.from_crs(_WGS84_CRS, dem.crs, always_xy=True)


def _to_dem_coordinates(
    transformer: Transformer,
    lon: float,
    lat: float,
) -> tuple[float, float]:
    x_coord, y_coord = transformer.transform(lon, lat, errcheck=True)
    if not isfinite(x_coord) or not isfinite(y_coord):
        raise ValueError("coordinate transformation returned a non-finite value")
    return float(x_coord), float(y_coord)


def _to_wgs84_transformer(dem: DemGrid) -> Transformer:
    return Transformer.from_crs(dem.crs, _WGS84_CRS, always_xy=True)


def _pixel_center_wgs84(
    dem: DemGrid,
    transformer: Transformer,
    row: int,
    col: int,
) -> tuple[float, float]:
    x_coord, y_coord = xy(dem.transform, row, col, offset="center")
    lon, lat = transformer.transform(x_coord, y_coord, errcheck=True)
    if not isfinite(lon) or not isfinite(lat):
        raise ValueError("pixel center transformation returned a non-finite value")
    return float(lon), float(lat)


def _local_slope_aspect(dem: DemGrid, row: int, col: int) -> tuple[float, float]:
    transformer = _to_wgs84_transformer(dem)
    center_lon, center_lat = _pixel_center_wgs84(dem, transformer, row, col)
    points = []
    elevations = []

    row_start = max(0, row - 1)
    row_stop = min(dem.shape[0], row + 2)
    col_start = max(0, col - 1)
    col_stop = min(dem.shape[1], col + 2)
    for neighbour_row in range(row_start, row_stop):
        for neighbour_col in range(col_start, col_stop):
            if dem.mask[neighbour_row, neighbour_col]:
                continue
            elevation = float(dem.elevation[neighbour_row, neighbour_col])
            if not isfinite(elevation):
                continue
            lon, lat = _pixel_center_wgs84(
                dem,
                transformer,
                neighbour_row,
                neighbour_col,
            )
            azimuth, _, distance = _WGS84_GEOD.inv(center_lon, center_lat, lon, lat)
            if not isfinite(distance) or not isfinite(azimuth):
                continue
            azimuth_rad = radians(azimuth)
            east = sin(azimuth_rad) * distance
            north = cos(azimuth_rad) * distance
            points.append((east, north, 1.0))
            elevations.append(elevation)

    if len(points) < 3:
        return 0.0, 0.0

    design = np.asarray(points, dtype=float)
    if np.linalg.matrix_rank(design) < 3:
        return 0.0, 0.0
    coefficients, *_ = np.linalg.lstsq(
        design,
        np.asarray(elevations, dtype=float),
        rcond=None,
    )
    east_gradient, north_gradient = map(float, coefficients[:2])
    if not isfinite(east_gradient) or not isfinite(north_gradient):
        return 0.0, 0.0

    rise = hypot(east_gradient, north_gradient)
    if rise <= np.finfo(float).eps:
        return 0.0, 0.0
    slope = min(90.0, max(0.0, degrees(atan(rise))))
    aspect = (degrees(atan2(-east_gradient, -north_gradient)) + 360.0) % 360.0
    return slope, aspect


def query_dem_at_point(dem_data, lon: float, lat: float) -> TerrainSample:
    if dem_data is None:
        return _default_sample("missing_dem")
    if not isinstance(dem_data, DemGrid):
        return _default_sample("missing_georeference")

    try:
        longitude = float(lon)
        latitude = float(lat)
    except (TypeError, ValueError):
        return _default_sample("invalid_coordinate")
    if (
        not isfinite(longitude)
        or not isfinite(latitude)
        or not -180.0 <= longitude <= 180.0
        or not -90.0 <= latitude <= 90.0
    ):
        return _default_sample("invalid_coordinate")
    if not _has_valid_georeference(dem_data):
        return _default_sample("missing_georeference")

    try:
        transformer = _to_dem_transformer(dem_data)
    except (CRSError, ProjError):
        return _default_sample("missing_georeference")

    try:
        x_coord, y_coord = _to_dem_coordinates(transformer, longitude, latitude)
        row, col = rowcol(dem_data.transform, x_coord, y_coord, op=np.floor)
        row = int(row)
        col = int(col)
    except (CRSError, ProjError, TypeError, ValueError, OverflowError):
        return _default_sample("invalid_coordinate")

    rows, cols = dem_data.shape
    if row < 0 or col < 0 or row >= rows or col >= cols:
        return _default_sample("out_of_bounds")
    if dem_data.mask[row, col]:
        return _default_sample("nodata")

    elevation = float(dem_data.elevation[row, col])
    if not isfinite(elevation):
        return _default_sample("nodata")
    try:
        slope, aspect = _local_slope_aspect(dem_data, row, col)
    except (
        CRSError,
        ProjError,
        TypeError,
        ValueError,
        OverflowError,
        np.linalg.LinAlgError,
    ):
        slope, aspect = 0.0, 0.0
    return TerrainSample(
        slope=float(slope),
        aspect=float(aspect),
        elevation=elevation,
        source="measured",
        reason=None,
    )


def _normalize_tower_id(value) -> str:
    if pd.isna(value):
        raise ValueError("invalid tower id")
    text = str(value).strip()
    label_match = re.search(r"(?<![\d.])(\d+)\s*号\s*$", text)
    if label_match:
        return label_match.group(1)
    trailing_match = re.search(r"(?<![\d.])(\d+)\s*$", text)
    if trailing_match:
        return trailing_match.group(1)

    numeric = pd.to_numeric(text, errors="coerce")
    if pd.notna(numeric) and np.isfinite(numeric) and numeric >= 0:
        integer = int(numeric)
        if numeric == integer:
            return str(integer)
    raise ValueError("invalid tower id")


def _rewind(file_or_path):
    if isinstance(file_or_path, (str, bytes, os.PathLike)):
        return
    try:
        file_or_path.seek(0)
    except (AttributeError, OSError, ValueError):
        pass


def load_tower_coordinates(excel_file, tower_nums=None) -> dict[str, dict[str, float]]:
    _rewind(excel_file)
    preview = pd.read_excel(excel_file, header=None, nrows=20)
    header_row = 0
    for index, row in preview.iterrows():
        row_text = " ".join(str(value) for value in row.values)
        if "运行编号" in row_text or "设备名称" in row_text:
            header_row = int(index)
            break

    _rewind(excel_file)
    frame = pd.read_excel(excel_file, header=header_row)
    frame.columns = [str(column).strip() for column in frame.columns]
    name_col = next(
        (column for column in frame.columns if "运行编号" in column or "设备名称" in column),
        None,
    )
    lon_col = next(
        (column for column in frame.columns if "经度" in column or "X坐标" in column),
        None,
    )
    lat_col = next(
        (column for column in frame.columns if "纬度" in column or "Y坐标" in column),
        None,
    )
    if not all((name_col, lon_col, lat_col)):
        raise MissingTowerColumnsError(
            f"关键列未找到！检测到的列: {frame.columns.tolist()}"
        )

    output = {}
    observed_ids = set()
    conflicting_ids = set()
    for _, row in frame.iterrows():
        try:
            tower_id = _normalize_tower_id(row[name_col])
            longitude = float(row[lon_col])
            latitude = float(row[lat_col])
        except (TypeError, ValueError):
            continue
        if not isfinite(longitude) or not isfinite(latitude):
            continue
        observed_ids.add(tower_id)
        if tower_id in conflicting_ids:
            continue
        coordinates = {"lon": longitude, "lat": latitude}
        if tower_id not in output:
            output[tower_id] = coordinates
        elif output[tower_id] != coordinates:
            output.pop(tower_id)
            conflicting_ids.add(tower_id)
    return _filter_tower_coordinates(
        output,
        tower_nums,
        numeric_candidate_ids=observed_ids,
    )


def _is_legacy_numeric_position(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (Integral, np.integer)):
        return int(value) >= 0
    if not isinstance(value, (Real, np.floating)):
        return False
    numeric = float(value)
    return isfinite(numeric) and numeric >= 0 and numeric.is_integer()


def _filter_tower_coordinates(
    coordinates: Mapping,
    tower_nums,
    *,
    numeric_candidate_ids=None,
) -> dict:
    if tower_nums is None:
        return dict(coordinates)

    if numeric_candidate_ids is None:
        numeric_candidate_ids = coordinates

    selected_ids = set()
    for tower_num in tower_nums:
        if _is_legacy_numeric_position(tower_num):
            number = int(tower_num)
            matches = [
                tower_id
                for tower_id in numeric_candidate_ids
                if int(tower_id) == number
            ]
            if len(matches) == 1 and matches[0] in coordinates:
                selected_ids.add(matches[0])
            continue
        try:
            tower_id = _normalize_tower_id(tower_num)
        except (TypeError, ValueError):
            continue
        if tower_id in coordinates:
            selected_ids.add(tower_id)
    return {
        tower_id: coordinate
        for tower_id, coordinate in coordinates.items()
        if tower_id in selected_ids
    }


def _write_temporary_upload(content: bytes, suffix: str) -> str:
    path = None
    completed = False
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            path = temporary_file.name
            temporary_file.write(content)
        completed = True
        return path
    finally:
        if path is not None and not completed:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def load_terrain_upload_pair(
    dem_content: bytes,
    tower_content: bytes,
    *,
    dem_loader=load_dem_data,
    tower_loader=load_tower_coordinates,
) -> TerrainLoadAttempt:
    temporary_paths = []
    try:
        dem_path = _write_temporary_upload(dem_content, ".tif")
        temporary_paths.append(dem_path)
        tower_path = _write_temporary_upload(tower_content, ".xlsx")
        temporary_paths.append(tower_path)
        dem_data = dem_loader(dem_path)
        tower_coords = tower_loader(tower_path)
        return TerrainLoadAttempt(dem_data=dem_data, tower_coords=tower_coords)
    finally:
        for path in temporary_paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def commit_terrain_snapshot(
    state: MutableMapping,
    attempt: TerrainLoadAttempt,
) -> bool:
    if not attempt.success:
        return False
    state.update(
        {
            "dem_data": attempt.dem_data,
            "tower_coords": attempt.tower_coords,
        }
    )
    return True


def _find_coordinates(
    tower_coords: Mapping,
    tower_id: str,
    original_id,
    *,
    legacy_mode: bool,
):
    if legacy_mode:
        numeric_matches = []
        for coordinate_id, coordinates in tower_coords.items():
            try:
                coordinate_number = int(_normalize_tower_id(coordinate_id))
            except (TypeError, ValueError):
                continue
            if coordinate_number == int(original_id):
                numeric_matches.append(coordinates)
        return numeric_matches[0] if len(numeric_matches) == 1 else None

    if tower_id in tower_coords:
        return tower_coords[tower_id]
    try:
        if original_id in tower_coords:
            return tower_coords[original_id]
    except TypeError:
        pass
    for coordinate_id, coordinates in tower_coords.items():
        try:
            normalized_coordinate_id = _normalize_tower_id(coordinate_id)
        except (TypeError, ValueError):
            continue
        if normalized_coordinate_id == tower_id:
            return coordinates
    return None


def build_terrain_lookup(
    dem_data,
    tower_coords: Mapping,
    weather_positions: list,
) -> dict:
    positions = list(weather_positions)
    legacy_mode = bool(positions) and all(
        _is_legacy_numeric_position(value) for value in positions
    )
    output = {}

    for index, position in enumerate(positions):
        try:
            tower_id = _normalize_tower_id(position)
        except (TypeError, ValueError):
            tower_id = str(position).strip()
        output_key = index if legacy_mode else tower_id

        if dem_data is None:
            output[output_key] = _default_sample("missing_dem")
            continue
        coordinates = _find_coordinates(
            tower_coords or {},
            tower_id,
            position,
            legacy_mode=legacy_mode,
        )
        if not coordinates or "lon" not in coordinates or "lat" not in coordinates:
            output[output_key] = _default_sample("missing_coordinates")
            continue
        output[output_key] = query_dem_at_point(
            dem_data,
            coordinates["lon"],
            coordinates["lat"],
        )
    return output
