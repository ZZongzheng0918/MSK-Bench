"""Unit tests for model mass validation.
"""

import xml.etree.ElementTree as ET

import mujoco as mj
import pytest
from musclemimic_models import get_xml_path


def get_body_names_from_xml(xml_path):
    """
    Returns a list of body names defined directly in a given XML file.
    Does not recurse into includes.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    body_names = []
    for elem in root.iter("body"):
        name = elem.get("name")
        if name is not None:
            body_names.append(name)
    return body_names


def get_total_mass_for_body_list(model, body_names):
    """Calculate total mass for a list of body names in the model."""
    total = 0.0
    for name in body_names:
        try:
            body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        except Exception:
            pytest.fail(f"Body '{name}' not found in compiled model")
        total += model.body_mass[body_id]
    return total


def compute_mass_by_xml(full_body_xml, subfiles):
    """Compile model and compute mass for each component."""
    model = mj.MjModel.from_xml_path(str(full_body_xml))

    results = {}
    for label, xml in subfiles.items():
        body_names = get_body_names_from_xml(xml)
        mass = get_total_mass_for_body_list(model, body_names)
        results[label] = mass

    return results, model


@pytest.fixture
def model_paths():
    """Fixture providing paths to model XML files."""
    full_body_xml = get_xml_path("myofullbody")

    model_root = full_body_xml.parent.parent

    return {
        "subfiles": {
            "arm": model_root / "arm" / "assets" / "myoarm_bimanual_body.xml",
            "leg": model_root / "leg" / "assets" / "myolegs_chain.xml",
            "trunk": model_root / "torso" / "assets" / "myotorso_bimanual_chain.xml",
            "head": model_root / "head" / "assets" / "myohead_rigid_chain.xml",
        },
        "full_body": full_body_xml,
    }


@pytest.fixture
def collision_geom_paths():
    """Fixture providing paths to collision geometry asset files."""
    muscle_model_path = get_xml_path("myofullbody")

    return [
        muscle_model_path / "arm" / "assets" / "myoarm_bimanual_assets.xml",
        muscle_model_path / "leg" / "assets" / "myolegs_assets.xml",
        muscle_model_path / "torso" / "assets" / "myotorso_assets.xml",
    ]


class TestModelMass:
    """Test suite for model mass validation."""

    def test_component_masses_within_expected_ranges(self, model_paths):
        """Test that component masses are within expected ranges.

        This ensures that collision geometries have mass="0" set correctly
        and that body masses are loaded as expected.
        """
        results, model = compute_mass_by_xml(
            full_body_xml=model_paths["full_body"], subfiles=model_paths["subfiles"]
        )
        # These values are based on the correct model with mass="0" on collision geoms
        expected_ranges = {
            "arm": (7.5, 11.0),
            "leg": (32.0, 43.0),
            "trunk": (29.0, 39.0),
            "head": (2.0, 3.0),
        }

        for component, (min_mass, max_mass) in expected_ranges.items():
            actual_mass = results[component]
            assert min_mass <= actual_mass <= max_mass, (
                f"{component} mass {actual_mass:.3f} kg is outside expected range "
                f"[{min_mass}, {max_mass}] kg. This may indicate missing mass=\"0\" "
                f"attributes on collision geometries or other mass computation issues."
            )

    def test_collision_geoms_have_zero_mass(self, collision_geom_paths):
        """Test that collision geometries have mass="0" attribute set.
        """
        for xml_file in collision_geom_paths:
            if not xml_file.exists():
                pytest.skip(f"File {xml_file} not found")

            tree = ET.parse(str(xml_file))
            root = tree.getroot()

            for default_elem in root.iter("default"):
                class_name = default_elem.get("class", "")
                if "coll" in class_name.lower():
                    # Check if any geom in this default has mass attribute
                    for geom_elem in default_elem.iter("geom"):
                        mass_attr = geom_elem.get("mass")
                        assert mass_attr is not None, (
                            f"Collision geom in class '{class_name}' in {xml_file.name} "
                            f"is missing mass attribute. This will cause MuJoCo to infer "
                            f"mass from geometry, resulting in incorrect total mass."
                        )
                        assert mass_attr == "0", (
                            f"Collision geom in class '{class_name}' in {xml_file.name} "
                            f"has mass='{mass_attr}' but should be mass='0'. "
                            f"Collision geometries should not contribute to model mass."
                        )

    def test_all_model_files_exist(self, model_paths):
        """Test that all required model XML files exist."""
        full_body_path = model_paths["full_body"]
        assert full_body_path.exists(), f"Full body model not found at {full_body_path}"

        for component, path in model_paths["subfiles"].items():
            assert path.exists(), f"{component} model not found at {path}"
