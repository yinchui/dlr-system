from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
from affine import Affine
from pyproj import Transformer
from pyproj.exceptions import ProjError
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

import modules.terrain as terrain


def write_test_geotiff(
    tmp_path: Path,
    values=None,
    *,
    crs="EPSG:4326",
    transform=None,
    nodata=None,
    dataset_mask=None,
    name="dem.tif",
) -> Path:
    if values is None:
        values = np.array([[100.0, 110.0], [120.0, 130.0]], dtype="float32")
    values = np.asarray(values)
    if transform is None:
        transform = from_origin(120.0, 50.0, 0.01, 0.01)

    path = tmp_path / name
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=values.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dataset:
        dataset.write(values, 1)
        if dataset_mask is not None:
            dataset.write_mask(np.asarray(dataset_mask, dtype="uint8"))
    return path


def test_geotiff_query_uses_crs_transform_and_affine(tmp_path):
    tif_path = write_test_geotiff(
        tmp_path,
        values=np.array([[100.0, 110.0], [120.0, 130.0]], dtype="float32"),
        crs="EPSG:4326",
        transform=from_origin(120.0, 50.0, 0.01, 0.01),
    )

    dem = terrain.load_dem_data(tif_path)
    result = terrain.query_dem_at_point(dem, lon=120.005, lat=49.995)

    assert result["elevation"] == pytest.approx(100.0)
    assert result["source"] == "measured"
    assert result["reason"] is None


def test_out_of_bounds_is_default_not_edge_pixel(tmp_path):
    dem = terrain.load_dem_data(write_test_geotiff(tmp_path))

    result = terrain.query_dem_at_point(dem, lon=0.0, lat=0.0)

    assert result["elevation"] == 1000.0
    assert result["source"] == "default"
    assert result["reason"] == "out_of_bounds"


@pytest.mark.parametrize(
    ("lon", "lat"),
    [
        (120.02, 49.995),
        (120.005, 49.98),
    ],
)
def test_right_and_bottom_boundaries_are_not_clamped(tmp_path, lon, lat):
    dem = terrain.load_dem_data(write_test_geotiff(tmp_path))

    result = terrain.query_dem_at_point(dem, lon=lon, lat=lat)

    assert result["source"] == "default"
    assert result["reason"] == "out_of_bounds"


def test_projected_crs_is_transformed_before_affine_lookup(tmp_path):
    values = np.tile(np.array([0.0, 10.0, 20.0], dtype="float32"), (3, 1))
    dem = terrain.load_dem_data(
        write_test_geotiff(
            tmp_path,
            values,
            crs="EPSG:3857",
            transform=from_origin(0.0, 30.0, 10.0, 10.0),
        )
    )
    lon, lat = Transformer.from_crs(3857, 4326, always_xy=True).transform(15.0, 15.0)

    result = terrain.query_dem_at_point(dem, lon=lon, lat=lat)

    assert result["elevation"] == pytest.approx(10.0)
    assert result["slope"] == pytest.approx(45.0, abs=0.2)
    assert result["aspect"] == pytest.approx(270.0, abs=0.2)
    assert result["source"] == "measured"


@pytest.mark.parametrize("coordinate", [(float("nan"), 49.0), (120.0, float("inf"))])
def test_non_finite_coordinates_return_default(tmp_path, coordinate):
    dem = terrain.load_dem_data(write_test_geotiff(tmp_path))

    result = terrain.query_dem_at_point(dem, lon=coordinate[0], lat=coordinate[1])

    assert result["source"] == "default"
    assert result["reason"] == "invalid_coordinate"


def test_nodata_pixel_returns_default(tmp_path):
    values = np.array([[-9999.0, 110.0], [120.0, 130.0]], dtype="float32")
    dem = terrain.load_dem_data(write_test_geotiff(tmp_path, values, nodata=-9999.0))

    result = terrain.query_dem_at_point(dem, lon=120.005, lat=49.995)

    assert result["elevation"] == 1000.0
    assert result["source"] == "default"
    assert result["reason"] == "nodata"


def test_dataset_mask_pixel_returns_default(tmp_path):
    mask = np.array([[0, 255], [255, 255]], dtype="uint8")
    dem = terrain.load_dem_data(
        write_test_geotiff(tmp_path, dataset_mask=mask)
    )

    result = terrain.query_dem_at_point(dem, lon=120.005, lat=49.995)

    assert result["source"] == "default"
    assert result["reason"] == "nodata"


def test_missing_crs_returns_default_without_inventing_coordinates(tmp_path):
    dem = terrain.load_dem_data(write_test_geotiff(tmp_path, crs=None))

    result = terrain.query_dem_at_point(dem, lon=120.005, lat=49.995)

    assert result["source"] == "default"
    assert result["reason"] == "missing_georeference"


def test_local_crs_without_wgs84_transform_returns_missing_georeference(tmp_path):
    local_crs = (
        'LOCAL_CS["Plant grid",LOCAL_DATUM["Plant datum",32767],'
        'UNIT["metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    )
    dem = terrain.load_dem_data(
        write_test_geotiff(
            tmp_path,
            crs=local_crs,
            transform=from_origin(0.0, 20.0, 10.0, 10.0),
        )
    )

    result = terrain.query_dem_at_point(dem, lon=0.0, lat=0.0)

    assert result["elevation"] == 1000.0
    assert result["source"] == "default"
    assert result["reason"] == "missing_georeference"


def test_reverse_transform_failure_keeps_measured_center_elevation(tmp_path, monkeypatch):
    dem = terrain.load_dem_data(write_test_geotiff(tmp_path))
    original_from_crs = terrain.Transformer.from_crs

    def from_crs(crs_from, crs_to, **kwargs):
        if crs_from is dem.crs and crs_to == "EPSG:4326":
            raise ProjError("reverse transform unavailable")
        return original_from_crs(crs_from, crs_to, **kwargs)

    monkeypatch.setattr(terrain.Transformer, "from_crs", staticmethod(from_crs))

    result = terrain.query_dem_at_point(dem, lon=120.005, lat=49.995)

    assert result["elevation"] == pytest.approx(100.0)
    assert result["slope"] == 0.0
    assert result["aspect"] == 0.0
    assert result["source"] == "measured"
    assert result["reason"] is None


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_identity_transform_returns_missing_georeference(tmp_path):
    dem = terrain.load_dem_data(
        write_test_geotiff(tmp_path, crs="EPSG:4326", transform=Affine.identity())
    )

    result = terrain.query_dem_at_point(dem, lon=0.5, lat=0.5)

    assert result["source"] == "default"
    assert result["reason"] == "missing_georeference"


def test_geographic_raster_slope_uses_metre_distances(tmp_path):
    values = np.tile(np.array([0.0, 10.0, 20.0], dtype="float32"), (3, 1))
    dem = terrain.load_dem_data(
        write_test_geotiff(
            tmp_path,
            values,
            transform=from_origin(120.0, 50.0, 0.01, 0.01),
        )
    )

    result = terrain.query_dem_at_point(dem, lon=120.015, lat=49.985)

    assert np.isfinite(result["slope"])
    assert 0.0 < result["slope"] < 5.0
    assert result["aspect"] == pytest.approx(270.0, abs=0.5)


def test_insufficient_valid_neighbours_keeps_measured_center_with_flat_fallback(tmp_path):
    values = np.arange(9, dtype="float32").reshape(3, 3)
    mask = np.zeros((3, 3), dtype="uint8")
    mask[1, 1] = 255
    dem = terrain.load_dem_data(
        write_test_geotiff(tmp_path, values, dataset_mask=mask)
    )

    result = terrain.query_dem_at_point(dem, lon=120.015, lat=49.985)

    assert result["elevation"] == pytest.approx(4.0)
    assert result["slope"] == 0.0
    assert result["aspect"] == 0.0
    assert result["source"] == "measured"


def test_dem_grid_owns_read_only_arrays_and_survives_dataset_close(tmp_path):
    path = write_test_geotiff(tmp_path)
    with rasterio.open(path) as dataset:
        dem = terrain.load_dem_data(dataset)
    path.unlink()

    assert isinstance(dem, terrain.DemGrid)
    assert dem.crs.to_epsg() == 4326
    assert dem.shape == (2, 2)
    assert dem.bounds.left == pytest.approx(120.0)
    assert dem.elevation.flags.writeable is False
    assert dem.mask.flags.writeable is False
    with pytest.raises(ValueError):
        dem.elevation[0, 0] = 999.0
    with pytest.raises(FrozenInstanceError):
        dem.shape = (1, 1)

    result = terrain.query_dem_at_point(dem, lon=120.005, lat=49.995)
    assert result["elevation"] == pytest.approx(100.0)


def test_load_dem_accepts_memory_file_and_copies_first_band():
    values = np.array([[25.0, 30.0], [35.0, 40.0]], dtype="float32")
    with MemoryFile() as memory_file:
        with memory_file.open(
            driver="GTiff",
            height=2,
            width=2,
            count=2,
            dtype=values.dtype,
            crs="EPSG:4326",
            transform=from_origin(120.0, 50.0, 0.01, 0.01),
        ) as dataset:
            dataset.write(values, 1)
            dataset.write(values + 1000.0, 2)
        dem = terrain.load_dem_data(memory_file)

    result = terrain.query_dem_at_point(dem, lon=120.005, lat=49.995)
    assert result["elevation"] == pytest.approx(25.0)


@pytest.mark.parametrize(
    ("name_column", "lon_column", "lat_column"),
    [
        ("运行编号", "经度", "纬度"),
        ("设备名称", "X坐标", "Y坐标"),
    ],
)
def test_load_tower_coordinates_detects_header_and_preserves_leading_zeros(
    tmp_path,
    name_column,
    lon_column,
    lat_column,
):
    path = tmp_path / "towers.xlsx"
    frame = pd.DataFrame(
        {
            name_column: ["500kV林彦一线001号", "500kV林彦一线002号", "003号", "无编号"],
            lon_column: [120.005, 120.015, float("inf"), 120.0],
            lat_column: [49.995, 49.985, 49.975, 49.0],
        }
    )
    frame.to_excel(path, index=False, startrow=2)

    result = terrain.load_tower_coordinates(path, tower_nums=["001", "002", "003"])

    assert list(result) == ["001", "002"]
    assert result["001"] == {"lon": 120.005, "lat": 49.995}


def test_canonical_terrain_lookup_uses_tower_id_keys(tmp_path):
    dem = terrain.load_dem_data(write_test_geotiff(tmp_path))

    result = terrain.build_terrain_lookup(
        dem,
        {
            "001": {"lon": 120.005, "lat": 49.995},
            "1": {"lon": 0.0, "lat": 0.0},
        },
        ["001"],
    )

    assert list(result) == ["001"]
    assert result["001"]["elevation"] == pytest.approx(100.0)
    assert result["001"]["source"] == "measured"


def test_canonical_lookup_without_dem_keeps_string_key_and_reason():
    result = terrain.build_terrain_lookup(
        None,
        {"001": {"lon": 120.0, "lat": 49.0}},
        ["001"],
    )

    assert list(result) == ["001"]
    assert result["001"]["source"] == "default"
    assert result["001"]["reason"] == "missing_dem"


def test_canonical_lookup_does_not_merge_distinct_leading_zero_id(tmp_path):
    dem = terrain.load_dem_data(write_test_geotiff(tmp_path))

    result = terrain.build_terrain_lookup(
        dem,
        {"1": {"lon": 120.005, "lat": 49.995}},
        ["001"],
    )

    assert list(result) == ["001"]
    assert result["001"]["source"] == "default"
    assert result["001"]["reason"] == "missing_coordinates"


def test_legacy_integer_lookup_retains_array_index_keys():
    result = terrain.build_terrain_lookup(None, {}, [36, 372])

    assert list(result) == [0, 1]
    assert all(sample["reason"] == "missing_dem" for sample in result.values())


def test_legacy_integer_lookup_matches_unique_zero_padded_loader_id(tmp_path):
    tower_path = tmp_path / "legacy-tower.xlsx"
    pd.DataFrame(
        {
            "运行编号": ["500kV林彦一线036号"],
            "经度": [120.005],
            "纬度": [49.995],
        }
    ).to_excel(tower_path, index=False)
    coordinates = terrain.load_tower_coordinates(tower_path)
    dem = terrain.load_dem_data(write_test_geotiff(tmp_path))

    result = terrain.build_terrain_lookup(dem, coordinates, [36])

    assert list(coordinates) == ["036"]
    assert list(result) == [0]
    assert result[0]["source"] == "measured"
    assert result[0]["elevation"] == pytest.approx(100.0)


def test_legacy_integer_lookup_rejects_ambiguous_numeric_coordinate_ids(tmp_path):
    dem = terrain.load_dem_data(write_test_geotiff(tmp_path))

    result = terrain.build_terrain_lookup(
        dem,
        {
            "36": {"lon": 120.005, "lat": 49.995},
            "036": {"lon": 120.015, "lat": 49.985},
        },
        [36],
    )

    assert result[0]["source"] == "default"
    assert result[0]["reason"] == "missing_coordinates"


def test_terrain_sample_supports_mapping_access(tmp_path):
    dem = terrain.load_dem_data(write_test_geotiff(tmp_path))

    sample = terrain.query_dem_at_point(dem, lon=120.005, lat=49.995)

    assert isinstance(sample, terrain.TerrainSample)
    assert sample["slope"] == sample.get("slope")
    assert dict(sample)["source"] == "measured"
