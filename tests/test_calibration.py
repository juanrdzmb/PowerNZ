import pytest

from calibration import create_calibration_from_plate_diameter


def test_create_calibration_from_plate_diameter() -> None:
    calibration = create_calibration_from_plate_diameter(150.0)

    assert calibration.plate_diameter_pixels == 150.0
    assert calibration.meters_per_pixel == pytest.approx(0.003)
    assert calibration.pixels_per_meter == pytest.approx(333.333333)
    assert calibration.pixels_to_meters(10.0) == pytest.approx(0.03)
    assert calibration.meters_to_pixels(0.45) == pytest.approx(150.0)


def test_create_calibration_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        create_calibration_from_plate_diameter(0.0)

    with pytest.raises(ValueError):
        create_calibration_from_plate_diameter(100.0, plate_diameter_meters=0.0)
