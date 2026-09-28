"""Quick manual check: send the worked mismatch example to /analyze and save the scorecard.

Not a pytest test; it is used to capture the worked example for the report. The result goes to
scripts/_worked_example.json rather than the terminal so it can be read back exactly.

Run (from anywhere):  python backend/scripts/smoke_analyze.py
"""

import json
import sys
from io import BytesIO
from pathlib import Path

# Make the backend package importable however this script is launched.
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.main import app  # noqa: E402


def main() -> None:
    client = TestClient(app)
    buf = BytesIO()
    Image.new("RGB", (120, 60), "white").save(buf, format="PNG")

    # The history lookup matches image content, not the filename, so this plain white image
    # should not match anything in the index.
    resp = client.post(
        "/analyze",
        files={"image": ("flood_recycled_2019.jpg", buf.getvalue(), "image/png")},
        data={
            "caption": "URGENT: massive flood devastating the city RIGHT NOW, "
            "share before they delete it!"
        },
    )
    result = {"http_status": resp.status_code, "scorecard": resp.json()}
    out_path = _BACKEND_DIR / "scripts" / "_worked_example.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
