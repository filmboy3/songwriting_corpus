import json
import instaloader
import pytesseract
from pathlib import Path
from PIL import Image
from datetime import datetime
import re
import os
import time
import logging
from tqdm import tqdm
import signal
import sys

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("instagram_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("instagram_scraper")

# --- Settings ---
USERNAME = "andreastolpeofficial"
POST_LIMIT = float('inf')  # Scrape all available posts
OUTPUT_DIR = Path("instagram_data")
OCR_DIR = OUTPUT_DIR / "ocr_text"
MARKDOWN_FILE = Path("andrea_stolpe_digest.md")
JSON_FILE = OUTPUT_DIR / f"{USERNAME}_data.json"
IMAGE_DIR = Path("instagram_images")
CORPUS_DIR = Path("songwriting_corpus") / "instagram"

# --- Setup ---
OUTPUT_DIR.mkdir(exist_ok=True)
OCR_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)
CORPUS_DIR.mkdir(exist_ok=True, parents=True)

# Checkpoint file for resuming scraping
CHECKPOINT_FILE = OUTPUT_DIR / f"{USERNAME}_checkpoint.json"

# Function to save checkpoint
def save_checkpoint(results, count, songwriting_count, skipped_count, error_count):
    checkpoint_data = {
        "results": results,
        "count": count,
        "songwriting_count": songwriting_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
        "timestamp": str(datetime.now())
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint_data, f)
    logger.info(f"Checkpoint saved with {len(results)} posts")

# Handle interruptions gracefully
def signal_handler(sig, frame):
    logger.warning("Received interrupt signal, saving checkpoint before exiting...")
    save_checkpoint(results, count, songwriting_count, skipped_count, error_count)
    print("\n⚠️ Script interrupted, checkpoint saved.")
    print(f"Resume later by loading checkpoint from {CHECKPOINT_FILE}")
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Initialize Instaloader
L = instaloader.Instaloader(download_pictures=True, download_videos=True,
                           download_video_thumbnails=True, download_comments=False,
                           save_metadata=False)

# Load session (recommended if you've done `instaloader --login your_user`)
try:
    L.load_session_from_file("jauntathan")  # Optional
    logger.info("Loaded existing session")
except Exception as e:
    logger.warning(f"Session loading failed: {e}")
    logger.info("Continuing without session (may hit rate limits)")

# --- Songwriting Keywords for Classification ---
SONGWRITING_KEYWORDS = [
    "songwriting", "lyrics", "melody", "chord", "harmony", "verse", "chorus", 
    "bridge", "hook", "rhyme", "composition", "songwriter", "writing", "song structure",
    "music theory", "creative process", "inspiration", "collaboration", "co-writing",
    "production", "arrangement", "recording", "demo", "publishing", "copyright",
    "performance", "artist", "singer", "musician", "guitar", "piano", "vocal",
    "technique", "craft", "storytelling", "emotion", "expression", "creativity"
]

def classify_songwriting_content(text):
    """Classify text based on songwriting relevance and extract topics"""
    if not text:
        return {"is_songwriting": False, "topics": [], "matched_keywords": [], "songwriting_score": 0}
    
    text_lower = text.lower()
    matched_keywords = [kw for kw in SONGWRITING_KEYWORDS if kw in text_lower]
    
    # Calculate a songwriting relevance score
    songwriting_score = len(matched_keywords)
    
    # Increase score for highly relevant keywords
    if any(kw in text_lower for kw in ["songwriting", "songwriter", "song writing"]):
        songwriting_score += 3
    
    topics = []
    # Check for specific songwriting topics
    if any(kw in text_lower for kw in ["lyric", "lyrics", "word", "words", "writing", "write"]):
        topics.append("lyrics")
    if any(kw in text_lower for kw in ["melody", "tune", "musical", "notes"]):
        topics.append("melody")
    if any(kw in text_lower for kw in ["chord", "harmony", "progression"]):
        topics.append("chords")
    if any(kw in text_lower for kw in ["structure", "form", "verse", "chorus", "bridge"]):
        topics.append("structure")
    if any(kw in text_lower for kw in ["inspiration", "idea", "creative", "creativity"]):
        topics.append("inspiration")
    if any(kw in text_lower for kw in ["process", "technique", "method", "approach", "craft"]):
        topics.append("process")
    if any(kw in text_lower for kw in ["collab", "co-write", "together", "partnership"]):
        topics.append("collaboration")
    if any(kw in text_lower for kw in ["production", "recording", "studio", "arrangement"]):
        topics.append("production")
    
    return {
        "is_songwriting": songwriting_score > 0,
        "topics": topics,
        "matched_keywords": matched_keywords,
        "songwriting_score": songwriting_score
    }

def extract_hashtags(text):
    """Extract hashtags from text"""
    if not text:
        return []
    return re.findall(r'#(\w+)', text)

def format_for_corpus(post_data):
    """Format post data for inclusion in the songwriting corpus"""
    topics = ", ".join(post_data.get("topics", ["general"]))
    hashtags = " ".join([f"#{tag}" for tag in post_data.get("hashtags", [])])
    
    formatted_text = f"""<INSTAGRAM>
<AUTHOR>{USERNAME}</AUTHOR>
<DATE>{post_data['date_utc']}</DATE>
<TOPICS>{topics}</TOPICS>
<URL>https://www.instagram.com/p/{post_data['shortcode']}/</URL>

<CONTENT>
{post_data['caption']}
</CONTENT>

<IMAGE_TEXT>
{post_data.get('ocr_text', '')}
</IMAGE_TEXT>

<HASHTAGS>
{hashtags}
</HASHTAGS>
</INSTAGRAM>
"""
    return formatted_text

# --- Scrape Posts ---
logger.info(f"Scraping posts from @{USERNAME}")
try:
    profile = instaloader.Profile.from_username(L.context, USERNAME)
    posts = profile.get_posts()
    logger.info(f"Found profile: {profile.full_name}")
    logger.info(f"Bio: {profile.biography[:100]}...")
    logger.info(f"Posts: {profile.mediacount}, Followers: {profile.followers}")
except Exception as e:
    logger.error(f"Error accessing profile: {e}")
    exit(1)

# Check for checkpoint
results = []
count = 0
songwriting_count = 0
skipped_count = 0
error_count = 0

# Load checkpoint if it exists
processed_shortcodes = set()
if CHECKPOINT_FILE.exists():
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            checkpoint_data = json.load(f)
            results = checkpoint_data.get("results", [])
            count = checkpoint_data.get("count", 0)
            songwriting_count = checkpoint_data.get("songwriting_count", 0)
            skipped_count = checkpoint_data.get("skipped_count", 0)
            error_count = checkpoint_data.get("error_count", 0)
            
            # Track processed shortcodes to avoid duplicates
            for post in results:
                processed_shortcodes.add(post.get("shortcode", ""))
            
        logger.info(f"Loaded checkpoint with {len(results)} posts from {checkpoint_data.get('timestamp')}")
        print(f"Resuming from checkpoint with {len(results)} posts already processed")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        print("Failed to load checkpoint, starting fresh")

# Create progress bar - use a reasonable initial value since we don't know total posts
with tqdm(desc="Scraping posts", unit="post") as pbar:
    for post in posts:
        if count >= POST_LIMIT:
            break
        
        # Skip if we've already processed this post
        if post.shortcode in processed_shortcodes:
            logger.debug(f"Skipping already processed post: {post.shortcode}")
            continue
            
        # Add delay to avoid rate limiting
        time.sleep(1.5)
        
        try:
            logger.info(f"Processing: {post.date_utc.date()} - {post.shortcode}")
            
            # Basic post data
            data = {
                "shortcode": post.shortcode,
                "caption": post.caption or "*No caption*",
                "date_utc": str(post.date_utc),
                "likes": post.likes,
                "comments": post.comments,
                "type": post.typename,
                "is_video": post.is_video,
                "url": f"https://www.instagram.com/p/{post.shortcode}/",
                "owner_username": USERNAME
            }
            
            # Extract hashtags
            data["hashtags"] = extract_hashtags(post.caption)
            
            # Classify caption content
            songwriting_info = classify_songwriting_content(post.caption)
            data.update(songwriting_info)
            
            # Download post content
            try:
                L.download_post(post, target=USERNAME)
            except Exception as dl_error:
                logger.warning(f"Download error for {post.shortcode}: {dl_error}")
                # Continue processing even if download fails
            
            # Try to find an image for OCR
            media_dir = OUTPUT_DIR / USERNAME
            image_candidates = list(media_dir.glob(f"{USERNAME}*_{post.shortcode}*.jpg"))
            if image_candidates:
                image_path = image_candidates[0]
                try:
                    # Save a copy of the image to our dedicated image directory
                    image_save_path = IMAGE_DIR / f"{post.shortcode}.jpg"
                    Image.open(image_path).save(image_save_path)
                    data["image_path"] = str(image_save_path)
                    
                    # Perform OCR
                    text = pytesseract.image_to_string(Image.open(image_path)).strip()
                    data["ocr_text"] = text or "*No image text detected*"
                    (OCR_DIR / f"{post.shortcode}.txt").write_text(text)
                    
                    # Also classify OCR text
                    ocr_songwriting_info = classify_songwriting_content(text)
                    if ocr_songwriting_info["is_songwriting"]:
                        data["is_songwriting"] = True
                        for topic in ocr_songwriting_info["topics"]:
                            if topic not in data["topics"]:
                                data["topics"].append(topic)
                        # Combine matched keywords
                        for kw in ocr_songwriting_info["matched_keywords"]:
                            if kw not in data["matched_keywords"]:
                                data["matched_keywords"].append(kw)
                except Exception as e:
                    logger.warning(f"OCR failed for {post.shortcode}: {e}")
                    data["ocr_text"] = f"*OCR failed: {e}*"
            else:
                data["ocr_text"] = "*No image found*"
            
            # Handle video
            if post.is_video:
                video_candidates = list(media_dir.glob(f"{USERNAME}*_{post.shortcode}*.mp4"))
                if video_candidates:
                    video_path = video_candidates[0]
                    # Get video metadata
                    import subprocess
                    try:
                        # Use ffprobe to get video duration
                        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                               '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
                        duration = float(subprocess.check_output(cmd).decode('utf-8').strip())
                        data["video_duration"] = duration
                    except Exception as e:
                        logger.warning(f"Error getting video duration: {e}")
                    
                    # Save video URL if available
                    try:
                        data["video_url"] = post.video_url
                    except Exception as e:
                        logger.warning(f"Error getting video URL: {e}")
            
            # Only count posts that are relevant to songwriting
            songwriting_hashtags = ["songwriting", "songwriter", "songwriters", "songwritingtips"]
            songwriting_relevant = (
                data["is_songwriting"] or 
                any(tag.lower() in songwriting_hashtags for tag in data["hashtags"]) or
                data.get("songwriting_score", 0) > 2
            )
            
            if songwriting_relevant:
                count += 1
                songwriting_count += 1
                results.append(data)
                logger.info(f"Added songwriting post ({count}/{POST_LIMIT}): {', '.join(data['topics'])}")
                pbar.update(1)
            else:
                skipped_count += 1
                logger.debug(f"Skipping non-songwriting post")
        
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing post {post.shortcode}: {e}")
            
            # If we hit too many errors, slow down
            if error_count > 5 and error_count % 5 == 0:
                logger.warning(f"Hit {error_count} errors, sleeping for 30 seconds")
                time.sleep(30)
                
            # If we hit too many consecutive errors, we might be rate limited
            if error_count > 20 and error_count % 20 == 0:
                logger.warning(f"Hit {error_count} errors, taking a longer break (2 minutes)")
                time.sleep(120)
        
        # Save checkpoint periodically
        if count % 25 == 0 and count > 0:
            save_checkpoint(results, count, songwriting_count, skipped_count, error_count)
            logger.info(f"Periodic checkpoint saved at {count} posts")

# --- Save Results ---
# Sort by date (newest first)
results.sort(key=lambda x: x["date_utc"], reverse=True)

# Save to JSON
with open(JSON_FILE, "w") as f:
    json.dump(results, f, indent=2)
logger.info(f"Saved {len(results)} posts to {JSON_FILE}")
logger.info(f"Summary: {songwriting_count} songwriting posts, {skipped_count} skipped, {error_count} errors")

# --- Generate Markdown ---
with open(MARKDOWN_FILE, "w") as out:
    out.write(f"# 🎵 Songwriting Resources: @{USERNAME}\n\n")
    out.write(f"## Overview\n\n")
    out.write(f"- Total posts analyzed: {len(results)}\n")
    
    # Count topics
    all_topics = {}
    for post in results:
        for topic in post.get("topics", []):
            all_topics[topic] = all_topics.get(topic, 0) + 1
    
    # Display topic statistics
    out.write("### Topics Covered\n\n")
    for topic, count in sorted(all_topics.items(), key=lambda x: x[1], reverse=True):
        out.write(f"- {topic.capitalize()}: {count} posts\n")
    
    out.write("\n## Posts\n\n")
    for post in results:
        date_fmt = datetime.fromisoformat(post["date_utc"]).strftime("%B %d, %Y")
        out.write(f"### 📅 {date_fmt}: {', '.join(post.get('topics', ['general']))}\n\n")
        out.write(f"**[View on Instagram](https://www.instagram.com/p/{post['shortcode']})**\n\n")
        out.write(f"#### 📝 Caption:\n{post['caption']}\n\n")
        out.write(f"#### 🔤 OCR from Image:\n```\n{post['ocr_text']}\n```\n")
        
        # Add hashtags section
        if post.get("hashtags"):
            out.write(f"#### #️⃣ Hashtags:\n")
            for tag in post["hashtags"]:
                out.write(f"#{tag} ")
            out.write("\n\n")
        
        out.write(f"\n---\n\n")

# --- Integrate with Songwriting Corpus ---
logger.info("Integrating with songwriting corpus...")
for post in results:
    # Format post for corpus
    formatted_text = format_for_corpus(post)
    
    # Create a sanitized filename
    date_str = datetime.fromisoformat(post["date_utc"]).strftime("%Y%m%d")
    filename = f"{date_str}_{USERNAME}_{post['shortcode']}.txt"
    
    # Save to corpus directory
    corpus_file = CORPUS_DIR / filename
    with open(corpus_file, "w") as f:
        f.write(formatted_text)

logger.info(f"Integrated {len(results)} posts into songwriting corpus at {CORPUS_DIR}")

print(f"\n✅ Finished processing Instagram data for @{USERNAME}")
print(f"📊 Summary:")
print(f"  - {songwriting_count} songwriting posts collected")
print(f"  - {skipped_count} non-songwriting posts skipped")
print(f"  - {error_count} errors encountered")
print(f"\n📁 Output files:")
print(f"  - JSON data: {JSON_FILE}")
print(f"  - Markdown digest: {MARKDOWN_FILE}")
print(f"  - Corpus files: {CORPUS_DIR}")
print(f"  - Images: {IMAGE_DIR}")
print(f"  - OCR text: {OCR_DIR}")
print(f"\n🔍 Next steps for preprocessing:")
print("  1. Review the generated digest to verify content quality")
print("  2. Run text normalization and cleaning on corpus files")
print("  3. Extract structured songwriting advice and techniques")
print("  4. Integrate with your existing songwriting corpus pipeline")