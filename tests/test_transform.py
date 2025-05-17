import numpy as np
import pytest

from metaarrays.transform import Centroid, FirstCoordinate


class TestFirstCoordinate:
    def test_transform(self) -> None:
        coords = np.array([10, 20, 30, 40, 50, 60, 70, 80])
        chunksizes = (3, 3, 2)
        transformer = FirstCoordinate()
        transformed_coords = transformer.transform(coords, chunksizes)
        assert np.array_equal(transformed_coords, np.array([10, 40, 70]))

    def test_invalid_length(self) -> None:
        coords = np.array([10, 20, 30, 40, 50])
        chunksizes = (3, 3, 2)
        transformer = FirstCoordinate()
        with pytest.raises(ValueError):
            transformer.transform(coords, chunksizes)


class TestCentroid:
    def test_transform(self) -> None:
        coords = np.array([10, 20, 30, 40, 50, 60, 70, 80])
        chunksizes = (3, 3, 2)
        transformer = Centroid()
        transformed_coords = transformer.transform(coords, chunksizes)
        assert np.array_equal(transformed_coords, np.array([20.0, 50.0, 75.0]))
