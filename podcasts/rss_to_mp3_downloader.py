#!/usr/bin/env python3
"""
RSS to MP3 Downloader

This script downloads podcast episodes from an RSS feed and saves them as MP3 files.
It handles errors gracefully and avoids re-downloading existing files.

Usage:
  python3 rss_to_mp3_downloader.py https://example.com/podcast/feed.xml --outdir mp3s
"""

import os
import re
import sys
import argparse
import requests
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("podcast_download.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rss_downloader")

def sanitize_filename(title: str) -> str:
    """Create a filesystem-safe filename."""
    # Replace problematic characters with underscores
    safe_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', title.strip())
    # Limit length to avoid filesystem issues
    return safe_name[:100]

def parse_rss(rss_url: str):
    """Yield (title, mp3_url, pub_date, episode_num) pairs from the RSS feed."""
    try:
        logger.info(f"Fetching RSS feed: {rss_url}")
        response = requests.get(rss_url, timeout=30)
        response.raise_for_status()
        
        # Handle potential XML parsing errors
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse RSS XML: {e}")
            return
        
        # Get podcast title for naming
        channel = root.find('channel')
        podcast_title = "podcast"
        if channel is not None and channel.find('title') is not None:
            podcast_title = channel.find('title').text.strip()
            podcast_title = sanitize_filename(podcast_title)
        
        # Process each item
        episode_count = 0
        for item in root.iter('item'):
            episode_count += 1
            
            # Extract episode details
            title_el = item.find('title')
            enclosure_el = item.find('enclosure')
            pub_date_el = item.find('pubDate')
            
            if title_el is None or enclosure_el is None:
                continue
                
            title = title_el.text.strip() if title_el.text else f"Episode_{episode_count}"
            mp3_url = enclosure_el.attrib.get('url')
            pub_date = pub_date_el.text if pub_date_el is not None else None
            
            # Check for audio URLs - either by extension or content type
            is_audio = False
            if mp3_url:
                # Check URL path (ignoring query parameters)
                url_path = mp3_url.split('?')[0]
                if url_path.lower().endswith(('.mp3', '.m4a', '.wav', '.aac')):
                    is_audio = True
                # Check if URL contains audio indicators
                elif any(x in mp3_url.lower() for x in ['audio', 'mp3', 'podcast', 'episode', 'track']):
                    is_audio = True
                # Check type attribute if available
                elif enclosure_el.get('type', '').startswith('audio/'):
                    is_audio = True
                    
            if is_audio:
                yield title, mp3_url, pub_date, episode_count, podcast_title
    
    except requests.RequestException as e:
        logger.error(f"Failed to fetch RSS feed: {e}")
        return

def download_mp3(mp3_url: str, dest_path: Path):
    """Download MP3 from URL to dest_path with progress bar."""
    try:
        # Make a HEAD request to get content length
        head_response = requests.head(mp3_url, timeout=10)
        file_size = int(head_response.headers.get('content-length', 0))
        
        # Stream the download with progress bar
        response = requests.get(mp3_url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    
    except Exception as e:
        logger.error(f"Download error: {e}")
        # Remove partial file if download failed
        if dest_path.exists():
            dest_path.unlink()
        return False

def main():
    parser = argparse.ArgumentParser(description="Download MP3s from a podcast RSS feed.")
    parser.add_argument("rss_url", help="The URL to the podcast RSS feed")
    parser.add_argument("--outdir", default="mp3s", help="Directory to save MP3 files")
    parser.add_argument("--limit", type=int, help="Limit number of episodes to download")
    parser.add_argument("--newest-first", action="store_true", help="Download newest episodes first")
    args = parser.parse_args()

    # Validate URL
    try:
        result = urlparse(args.rss_url)
        if not all([result.scheme, result.netloc]):
            logger.error(f"Invalid URL: {args.rss_url}")
            return 1
    except Exception:
        logger.error(f"Invalid URL: {args.rss_url}")
        return 1

    # Create output directory
    output_dir = Path(args.outdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading podcasts to: {output_dir}")
    
    # Get episodes from feed
    episodes = list(parse_rss(args.rss_url))
    
    if not episodes:
        logger.error("No episodes found in the RSS feed")
        return 1
    
    # Sort episodes if needed
    if args.newest_first:
        # Reverse the list to get newest first (assuming they're in chronological order)
        episodes.reverse()
    
    # Apply limit if specified
    if args.limit and args.limit > 0:
        episodes = episodes[:args.limit]
    
    logger.info(f"Found {len(episodes)} episodes to process")
    
    # Download episodes
    success_count = 0
    for title, mp3_url, pub_date, episode_num, podcast_title in episodes:
        # Create filename with podcast name, episode number and title
        safe_name = f"{podcast_title}_ep{episode_num:03d}_{sanitize_filename(title)}.mp3"
        mp3_path = output_dir / safe_name

        if mp3_path.exists():
            logger.info(f"Already downloaded: {mp3_path.name}")
            success_count += 1
            continue

        logger.info(f"Downloading: {title}")
        if download_mp3(mp3_url, mp3_path):
            logger.info(f"Successfully saved: {mp3_path.name}")
            success_count += 1
        else:
            logger.error(f"Failed to download: {title}")
    
    logger.info(f"Download complete. Successfully downloaded {success_count}/{len(episodes)} episodes.")
    return 0

if __name__ == "__main__":
    sys.exit(main())