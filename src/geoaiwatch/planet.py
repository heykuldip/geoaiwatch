"""Search, order and download PlanetScope scenes with Planet's Python SDK (planet 3).

    python -m geoaiwatch.planet search area.geojson 2018-04-01 2018-05-01 --min-clear 90
    python -m geoaiwatch.planet order area.geojson 20180402_014526_103d --name my-order --yes
    python -m geoaiwatch.planet download ORDER_ID data/planetscope

The key comes from PL_API_KEY. Orders are clipped to the area and delivered as
cloud-optimized GeoTIFFs, and they count against your account's allowance. Without --yes,
`order` prints what it would order and stops.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

ITEM_TYPE = "PSScene"
BUNDLE = "analytic_sr_udm2"
SR_ASSETS = ("ortho_analytic_4b_sr", "ortho_analytic_4b_xml", "ortho_udm2")
FINAL_STATES = {"success", "partial", "failed", "cancelled"}


def client():
    from planet import Planet
    return Planet()


def read_geometry(path):
    """The geometry in a GeoJSON file: a geometry, a feature, or a collection of one feature."""
    g = json.loads(Path(path).read_text())
    if g.get("type") == "FeatureCollection":
        n = len(g["features"])
        if n != 1:
            raise SystemExit(f"{path} holds {n} features; give a file with exactly one area")
        g = g["features"][0]
    return g.get("geometry", g)


def scene_filter(geometry, start, end, min_clear=None, max_cloud=None):
    """Standard-quality scenes over the area, acquired in [start, end), with SR and UDM2."""
    from planet import data_filter as F
    parts = [F.geometry_filter(geometry, relation="intersects"),
             F.date_range_filter("acquired", gte=start, lt=end),
             F.std_quality_filter(), F.permission_filter(),
             *[F.asset_filter([a]) for a in SR_ASSETS]]   # one asset per filter: a list means any
    if min_clear is not None:
        parts.append(F.range_filter("clear_percent", gte=min_clear))
    if max_cloud is not None:
        parts.append(F.range_filter("cloud_cover", lte=max_cloud))
    return F.and_filter(parts)


def search(geometry, start, end, min_clear=None, max_cloud=None, pl=None):
    pl = pl or client()
    items = pl.data.search([ITEM_TYPE], limit=0,                # 0: no cap; the default is 100
                           search_filter=scene_filter(geometry, start, end, min_clear, max_cloud))
    rows = [{"id": i["id"], "acquired": i["properties"]["acquired"],
             "clear_percent": i["properties"].get("clear_percent"),
             "cloud_cover": i["properties"].get("cloud_cover"),
             "instrument": i["properties"].get("instrument")} for i in items]
    return sorted(rows, key=lambda r: r["acquired"])


def order_request(geometry, item_ids, name):
    from planet import order_request as O
    return O.build_request(name=name,                            # tools by keyword: the third
                           products=[O.product(list(item_ids), BUNDLE, ITEM_TYPE)],  # positional
                           tools=[O.clip_tool(geometry), O.file_format_tool("COG")])  # is not tools


def order(geometry, item_ids, name, pl=None):
    pl = pl or client()
    return pl.orders.create_order(order_request(geometry, item_ids, name))["id"]


def wait(order_id, timeout_s=3600, poll_s=30, pl=None):
    pl = pl or client()
    t0 = time.time()
    while True:
        state = pl.orders.get_order(order_id)["state"]
        if state in FINAL_STATES or time.time() - t0 > timeout_s:
            return state
        time.sleep(poll_s)


def download(order_id, directory, pl=None):
    pl = pl or client()
    Path(directory).mkdir(parents=True, exist_ok=True)
    files = [Path(f) for f in pl.orders.download_order(order_id, directory=directory,
                                                       overwrite=False, progress_bar=True)]
    manifest = next((f for f in files if f.name == "manifest.json"), None)
    if manifest is not None:
        pl.orders.validate_checksum(manifest.parent, "MD5")
    return files


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m geoaiwatch.planet")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("area")
    s.add_argument("start", type=datetime.fromisoformat)
    s.add_argument("end", type=datetime.fromisoformat)
    s.add_argument("--min-clear", type=int)
    s.add_argument("--max-cloud", type=float, help="cloud_cover, from 0 to 1")
    o = sub.add_parser("order")
    o.add_argument("area")
    o.add_argument("item_ids", nargs="+")
    o.add_argument("--name", required=True)
    o.add_argument("--wait", action="store_true")
    o.add_argument("--yes", action="store_true", help="place the order; without it, only show it")
    d = sub.add_parser("download")
    d.add_argument("order_id")
    d.add_argument("directory")
    a = p.parse_args(argv)
    if a.cmd == "search":
        rows = search(read_geometry(a.area), a.start, a.end, a.min_clear, a.max_cloud)
        for r in rows:
            print(f"{r['id']}  {r['acquired'][:19]}  clear {r['clear_percent']}%  "
                  f"{r['instrument']}")
        print(f"{len(rows)} scenes")
    elif a.cmd == "order":
        geometry = read_geometry(a.area)
        print(f"order {a.name!r}: {len(a.item_ids)} {ITEM_TYPE} scenes, bundle {BUNDLE}, "
              f"clipped to the {geometry['type']} in {a.area}, delivered as COG")
        if not a.yes:
            print("not placed: add --yes to order; it counts against your account's allowance")
            return
        oid = order(geometry, a.item_ids, a.name)
        print(f"order {oid}")
        if a.wait:
            print(f"state {wait(oid)}")
    else:
        files = download(a.order_id, a.directory)
        print(f"{len(files)} files in {a.directory}")


if __name__ == "__main__":
    main()
