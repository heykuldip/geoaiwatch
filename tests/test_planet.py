"""The Planet module against a stand-in client, with fixtures shaped like the API's responses."""
import json
from datetime import datetime

import pytest

pytest.importorskip("planet")
from geoaiwatch import planet as PL  # noqa: E402

AREA = {"type": "Polygon", "coordinates": [[[119.40, 32.45], [119.44, 32.45], [119.44, 32.49],
                                            [119.40, 32.49], [119.40, 32.45]]]}
ITEMS = [{"id": "20180402_014526_103d", "properties": {"acquired": "2018-04-02T01:45:26Z",
          "clear_percent": 97, "cloud_cover": 0.01, "instrument": "PS2"}},
         {"id": "20180401_020000_0f02", "properties": {"acquired": "2018-04-01T02:00:00Z",
          "clear_percent": 88, "cloud_cover": 0.08, "instrument": "PS2.SD"}}]


class Orders:
    def __init__(self, states):
        self.states, self.created, self.checked = list(states), None, None

    def create_order(self, request):
        self.created = request
        return {"id": "order-1", "state": "queued"}

    def get_order(self, order_id):
        return {"id": order_id, "state": self.states.pop(0)}

    def download_order(self, order_id, directory, overwrite, progress_bar):
        return [f"{directory}/order-1/manifest.json", f"{directory}/order-1/scene_SR.tif"]

    def validate_checksum(self, directory, checksum):
        self.checked = (str(directory), checksum)


class Data:
    def __init__(self):
        self.kwargs = None

    def search(self, item_types, **kwargs):
        self.kwargs = dict(kwargs, item_types=item_types)
        return iter(ITEMS)


class Client:
    def __init__(self, states=("queued", "running", "success")):
        self.data, self.orders = Data(), Orders(states)


@pytest.fixture(autouse=True)
def offline_specs(monkeypatch):
    """The SDK checks item types and bundles against the live API; skip that offline."""
    import planet.order_request as OR
    import planet.specs as SP
    for mod in (SP, OR):
        monkeypatch.setattr(mod, "validate_item_type", lambda x: x, raising=False)
        monkeypatch.setattr(mod, "validate_bundle", lambda it, b, *a, **k: b, raising=False)


def _types(f):
    return [c["type"] for c in f["config"]]


def test_filter_asks_for_each_asset_separately():
    f = PL.scene_filter(AREA, datetime(2018, 4, 1), datetime(2018, 5, 1), min_clear=90)
    assert f["type"] == "AndFilter"
    assets = [c["config"] for c in f["config"] if c["type"] == "AssetFilter"]
    assert assets == [[a] for a in PL.SR_ASSETS]
    clear = [c for c in f["config"] if c["type"] == "RangeFilter"]
    assert clear == [{"type": "RangeFilter", "field_name": "clear_percent", "config": {"gte": 90}}]
    assert {"GeometryFilter", "DateRangeFilter", "PermissionFilter"} <= set(_types(f))


def test_search_has_no_cap_and_sorts_by_time():
    pl = Client()
    rows = PL.search(AREA, datetime(2018, 4, 1), datetime(2018, 5, 1), pl=pl)
    assert pl.data.kwargs["limit"] == 0 and pl.data.kwargs["item_types"] == ["PSScene"]
    assert [r["id"] for r in rows] == ["20180401_020000_0f02", "20180402_014526_103d"]
    assert rows[1]["clear_percent"] == 97


def test_order_is_clipped_and_delivered_as_cog():
    pl = Client()
    assert PL.order(AREA, ["20180402_014526_103d"], "test-order", pl=pl) == "order-1"
    req = pl.orders.created
    assert req["name"] == "test-order"
    assert req["products"] == [{"item_ids": ["20180402_014526_103d"], "item_type": "PSScene",
                                "product_bundle": "analytic_sr_udm2"}]
    tools = {k for t in req["tools"] for k in t}
    assert tools == {"clip", "file_format"}


def test_wait_stops_at_a_final_state():
    pl = Client(states=("queued", "running", "success"))
    assert PL.wait("order-1", poll_s=0, pl=pl) == "success"


def test_download_checks_the_manifest(tmp_path):
    pl = Client()
    files = PL.download("order-1", tmp_path, pl=pl)
    assert [f.name for f in files] == ["manifest.json", "scene_SR.tif"]
    assert pl.orders.checked == (str(tmp_path / "order-1"), "MD5")


def test_read_geometry_refuses_several_areas(tmp_path):
    two = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": AREA}] * 2}
    (tmp_path / "two.geojson").write_text(json.dumps(two))
    with pytest.raises(SystemExit, match="holds 2 features"):
        PL.read_geometry(tmp_path / "two.geojson")


def test_order_needs_yes(tmp_path, monkeypatch, capsys):
    (tmp_path / "area.geojson").write_text(json.dumps(AREA))
    pl = Client()
    monkeypatch.setattr(PL, "client", lambda: pl)
    args = ["order", str(tmp_path / "area.geojson"), "20180402_014526_103d", "--name", "t"]
    PL.main(args)
    assert "not placed" in capsys.readouterr().out and pl.orders.created is None
    PL.main(args + ["--yes"])
    assert "order order-1" in capsys.readouterr().out and pl.orders.created is not None
