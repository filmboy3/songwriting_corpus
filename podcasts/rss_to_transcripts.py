#!/usr/bin/env python3

import os
import re
import argparse
import requests
import xml.etree.ElementTree as ET
from pathlib import Path

def sanitize_filename(title: str) -> str:
    """Create a filesystem-safe filename."""
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', title.strip())[:100]

def parse_rss(rss_url: str):
    """Yield (title, mp3_url) pairs from the RSS feed."""
    response = requests.get(rss_url)
    response.raise_for_status()
    root = ET.fromstring(response.content)

    for item in root.iter('item'):
        title_el = item.find('title')
        enclosure_el = item.find('enclosure')
        if title_el is None or enclosure_el is None:
            continue
        title = title_el.text.strip()
        mp3_url = enclosure_el.attrib.get('url')
        if mp3_url:
            yield title, mp3_url

def download_mp3(mp3_url: str, dest_path: Path):
    """Download MP3 from URL to dest_path."""
    response = requests.get(mp3_url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(8192):
            f.write(chunk)

def main():
    parser = argparse.ArgumentParser(description="Download MP3s from a podcast RSS feed.")
    parser.add_argument("rss_url", help="The URL to the podcast RSS feed")
    parser.add_argument("--outdir", default="mp3s", help="Directory to save MP3 files")
    args = parser.parse_args()

    output_dir = Path(args.outdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for title, mp3_url in parse_rss(args.rss_url):
        safe_name = sanitize_filename(title) + ".mp3"
        mp3_path = output_dir / safe_name

        if mp3_path.exists():
            print(f"[✓] Already downloaded: {mp3_path.name}")
            continue

        print(f"[↓] Downloading: {title}")
        try:
            download_mp3(mp3_url, mp3_path)
            print(f"[💾] Saved: {mp3_path.name}")
        except Exception as e:
            print(f"[⚠️] Failed to download {title}: {e}")

if __name__ == "__main__":
    main()