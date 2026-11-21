import pytest
from scripts.export_ikea_scenes import select_objects

def test_selection_rejects_missing_food_before_export():
    with pytest.raises(ValueError, match="Unknown objects: tofu"):
        select_objects({"objects": [{"name": "cucumber"}]}, ["tofu"])

def test_selection_preserves_contract_order_and_accepts_all():
    objects = [{"name": name} for name in ("cucumber", "tofu", "banana")]
    assert select_objects({"objects": objects}) == objects
    assert select_objects({"objects": objects}, ["banana", "cucumber"]) == [objects[0], objects[2]]

def test_selection_rejects_empty_and_duplicate_contracts():
    for objects in ([], [{"name": "tofu"}, {"name": "tofu"}]):
        with pytest.raises(ValueError):
            select_objects({"objects": objects})
