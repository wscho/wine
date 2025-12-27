from __future__ import annotations

import zipfile
from pathlib import Path

import requests


GOOGLE_FONTS_ZIP_URL = "https://fonts.google.com/download?family=Noto%20Sans%20KR"

# 구글 폰트 ZIP이 HTML로 내려오는 환경(차단/리다이렉트 등) 대비: GitHub raw fallback
# google/fonts 저장소에는 고정 Regular/Bold TTF가 아닌 "가변 폰트" TTF가 제공됩니다.
# (아래 파일은 다운로드 확인됨)
GITHUB_VARIABLE_TTF = "https://raw.githubusercontent.com/google/fonts/main/ofl/notosanskr/NotoSansKR%5Bwght%5D.ttf"
FALLBACK_TTFS = {
    # 동일한 가변 폰트를 Regular/Bold로 저장(WordCloud/Matplotlib은 한글 폰트 파일만 있으면 됨)
    "NotoSansKR-Regular.ttf": GITHUB_VARIABLE_TTF,
    "NotoSansKR-Bold.ttf": GITHUB_VARIABLE_TTF,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _assets_fonts_dir() -> Path:
    return _repo_root() / "assets" / "fonts"


def download_and_extract_noto_sans_kr() -> None:
    out_dir = _assets_fonts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "*/*",
    }

    print(f"Downloading (zip): {GOOGLE_FONTS_ZIP_URL}")
    resp = requests.get(GOOGLE_FONTS_ZIP_URL, timeout=60, allow_redirects=True, headers=headers)
    resp.raise_for_status()

    # Google Fonts returns a zip
    zip_path = out_dir / "_NotoSansKR.zip"
    zip_path.write_bytes(resp.content)
    print(f"Saved: {zip_path}")

    # ZIP 시그니처( PK\x03\x04 / PK\x05\x06 / PK\x07\x08 ) 확인
    head = resp.content[:4]
    is_zip = head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06") or head.startswith(b"PK\x07\x08")
    if not is_zip:
        # 보통 HTML 오류 페이지가 내려옵니다.
        html_path = out_dir / "_NotoSansKR_response.html"
        try:
            html_path.write_bytes(resp.content)
            print(f"WARNING: response is not zip. Saved debug HTML: {html_path}")
        except Exception:
            print("WARNING: response is not zip (and failed to save debug HTML).")

        # fallback: GitHub raw TTF
        print("Falling back to GitHub raw TTF download...")
        for fn, url in FALLBACK_TTFS.items():
            dest = out_dir / fn
            print(f"Downloading: {url}")
            r = requests.get(url, timeout=60, allow_redirects=True, headers=headers)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"Wrote: {dest}")

        # cleanup zip
        try:
            zip_path.unlink()
        except Exception:
            pass

        # debug HTML은 용량/혼선을 줄이기 위해 기본적으로 삭제(필요 시 사용자가 남길 수 있음)
        try:
            html_path.unlink()
        except Exception:
            pass
        print("Done (fallback).")
        return

    wanted = {
        "NotoSansKR-Regular.ttf": None,
        "NotoSansKR-Bold.ttf": None,
    }

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        # find matching ttf in zip (often under static/)
        for target in list(wanted.keys()):
            match = next((n for n in names if n.endswith("/" + target) or n.endswith("\\" + target) or n.endswith(target)), None)
            if match:
                wanted[target] = match

        for target, member in wanted.items():
            if not member:
                print(f"WARNING: {target} not found in zip.")
                continue
            data = z.read(member)
            (out_dir / target).write_bytes(data)
            print(f"Wrote: {out_dir / target}  (from {member})")

    # cleanup zip
    try:
        zip_path.unlink()
        print(f"Removed temp zip: {zip_path}")
    except Exception:
        pass

    print("Done.")


if __name__ == "__main__":
    download_and_extract_noto_sans_kr()


