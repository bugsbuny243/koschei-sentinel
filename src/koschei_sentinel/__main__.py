from __future__ import annotations

import os

import uvicorn


def main() -> None:
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("koschei_sentinel.service:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
