"""Manual smoke test: run the canonical worked mismatch example through /analyze and print
the resulting scorecard. Not a pytest test — a developer/demo aid (and a way to capture the
worked example for the report). Run: python scripts/smoke_analyze.py
"""

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def main() -> None:
    client = TestClient(app)
    buf = BytesIO()
    Image.new("RGB", (120, 60), "white").save(buf, format="PNG")

    # The filename matches an entry in the cached reverse-image fixture, so the
    # recycled-context flag can fire (ADR-007).
    resp = client.post(
        "/analyze",
        files={"image": ("flood_recycled_2019.jpg", buf.getvalue(), "image/png")},
        data={
            "caption": "URGENT: massive flood devastating the city RIGHT NOW, "
            "share before they delete it!"
        },
    )
    print("HTTP", resp.status_code)
    card = resp.json()
    print("SUMMARY:", card["summary"])
    print("SCHEMA:", card.get("schema_version"))
    for f in card["flags"]:
        print(f"- [{f['status']}] {f['type']} (sev={f['severity']}, src={f['source']})")
        print(f"    {f['plain_explanation']}")
        print(f"    what_to_check: {f['what_to_check']}")


if __name__ == "__main__":
    main()
