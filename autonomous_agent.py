#!/usr/bin/env python3
"""
Autonomous OpenAI Agent for Storefront Discovery & Analysis
Uses the OpenAI Agents SDK to autonomously:
1. Discover venues in London
2. For each venue: fetch 8 Street View angles, analyze each
3. Save validated images, stop after finding 3 candidates
4. Fallback to Google Places photos if needed
"""

import asyncio
import json
import os
import time
import base64
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv

load_dotenv()

# Configuration
MAPS_API_KEY = os.getenv("MAPS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OUTPUT_DIR = Path("agent_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Run directory (always created)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_DIR / RUN_TIMESTAMP
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Image directories (always created)
STREETVIEW_DIR = RUN_DIR / "360_view"
GOOGLEPLACES_DIR = RUN_DIR / "google_places"
STREETVIEW_DIR.mkdir(parents=True, exist_ok=True)
GOOGLEPLACES_DIR.mkdir(parents=True, exist_ok=True)

# Candidates folder (created only if we have suitable images)
CANDIDATES_DIR = RUN_DIR / "candidates"

# Street View headings for full 360° coverage
HEADINGS = [0, 45, 90, 135, 180, 225, 270, 315]
HEADING_LABELS = ["N (0°)", "NE (45°)", "E (90°)", "SE (135°)", "S (180°)", "SW (225°)", "W (270°)", "NW (315°)"]

# London neighborhoods for diverse venue discovery
LONDON_NEIGHBORHOODS = [
    "Soho, London",
    "Shoreditch, London",
    "Camden, London",
    "Covent Garden, London",
    "Fitzrovia, London",
    "Clerkenwell, London",
    "Brick Lane, London",
    "Notting Hill, London",
    "King's Road Chelsea, London",
    "Carnaby Street, London"
]

# Global state
_candidates_found = 0
_analyzed_venues = []
_venues_with_images = set()  # Track unique venues that have suitable images
_venue_photos = {}  # Store photos metadata keyed by venue name
_all_analyzed_images = []  # Track ALL analyzed images with validation details


@function_tool
def search_venues(business_type: str, location: str = "London, UK") -> str:
    """Search for venues of a specific type"""
    global _venue_photos

    print(f"\n🔍 Searching {business_type}s...")

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": MAPS_API_KEY,
        "X-Goog-FieldMask": (
            "places.id,places.displayName,places.formattedAddress,"
            "places.rating,places.location,places.photos"
        )
    }
    body = {
        "textQuery": f"{business_type} in {location}",
        "maxResultCount": 8
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=10)
        resp.raise_for_status()
        places = resp.json().get("places", [])

        results = []
        for place in places:
            venue_name = place.get("displayName", {}).get("text")
            results.append({
                "name": venue_name,
                "address": place.get("formattedAddress"),
                "rating": place.get("rating"),
                "lat": place.get("location", {}).get("latitude"),
                "lng": place.get("location", {}).get("longitude"),
            })
            # Store photos metadata for fallback
            if venue_name:
                _venue_photos[venue_name] = place.get("photos", [])

        print(f"   ✅ Found {len(results)} venues")
        return json.dumps({"venues": results})
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return json.dumps({"venues": []})


@function_tool
def check_street_view(latitude: float, longitude: float) -> str:
    """Check if Street View is available at location"""
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{latitude},{longitude}", "key": MAPS_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        available = resp.json().get("status") == "OK"
        return json.dumps({"available": available})
    except Exception as e:
        return json.dumps({"available": False, "error": str(e)})


@function_tool
def analyze_venue_candidate(venue_name: str, address: str, latitude: float,
                           longitude: float, rating: Optional[float]) -> str:
    """
    Analyze a venue by fetching 8 Street View images at different headings.
    For each image, validate it. If suitable (score>=8, visible=YES, confidence>=70),
    save it and report success.
    If none of 8 angles work, try Google Places photos.
    """
    global _candidates_found

    if _candidates_found >= 3:
        return json.dumps({"status": "halt", "reason": "3 candidates already found"})

    print(f"\n📍 Analyzing: {venue_name}")
    print(f"   Address: {address}")
    print(f"   Rating: {rating}")

    # Try all 8 headings
    for heading, label in zip(HEADINGS, HEADING_LABELS):
        print(f"   📸 Trying {label}...")

        try:
            image_resp = requests.get(
                "https://maps.googleapis.com/maps/api/streetview",
                params={
                    "size": "600x400",
                    "location": f"{latitude},{longitude}",
                    "heading": heading,
                    "pitch": 0,
                    "fov": 90,
                    "key": MAPS_API_KEY
                },
                timeout=10
            )

            if not image_resp.ok:
                print("      ⚠️  Could not fetch")
                continue

            image_b64 = base64.b64encode(image_resp.content).decode()

            # Validate image
            validation = validate_storefront_image(image_b64, venue_name, address)

            is_suitable = (
                validation.get("suitable", False) and
                validation.get("confidence", 0) >= 70
            )

            print(f"      Storefront: {validation.get('is_storefront')}, "
                  f"Visible: {validation.get('entrance_visible')}, "
                  f"Bare: {validation.get('bare_enough')}, "
                  f"Suitable: {validation.get('suitable')}, "
                  f"Confidence: {validation.get('confidence')}%")

            # ALWAYS save analyzed images for debugging (organized by venue)
            venue_dir = STREETVIEW_DIR / venue_name.replace(' ', '_')
            venue_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{label.split()[0]}.jpg"
            filepath = venue_dir / filename
            filepath.write_bytes(image_resp.content)
            print(f"      💾 Saved to 360_view/{venue_name.replace(' ', '_')}/{filename}")

            # Log this analyzed image
            _all_analyzed_images.append({
                "type": "street_view",
                "venue": venue_name,
                "address": address,
                "file": str(filepath),
                "heading": label,
                "validation": validation,
                "is_suitable": is_suitable,
                "reason_rejected": None if is_suitable else f"Score {validation.get('confidence')}% < 70% or suitable={validation.get('suitable')}"
            })

            if is_suitable:
                # Check if this is a new venue
                is_new_venue = venue_name not in _venues_with_images

                # Only increment counter for new venues
                if is_new_venue:
                    _candidates_found += 1
                    _venues_with_images.add(venue_name)
                    # Create candidates folder on first suitable image
                    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

                # Also save to candidates folder
                candidates_venue_dir = CANDIDATES_DIR / venue_name.replace(' ', '_')
                candidates_venue_dir.mkdir(parents=True, exist_ok=True)
                candidates_filepath = candidates_venue_dir / filename
                candidates_filepath.write_bytes(image_resp.content)

                print("      ✅ SUITABLE! Also saved to candidates/")

                _analyzed_venues.append({
                    "name": venue_name,
                    "address": address,
                    "rating": rating,
                    "image": str(filepath),
                    "heading": label,
                    "validation": validation
                })

                if _candidates_found >= 3:
                    print("\n🏆 FOUND 3 CANDIDATES! Stopping search.")
                    return json.dumps({
                        "status": "success",
                        "candidates": _candidates_found,
                        "venue": venue_name,
                        "image": str(filepath)
                    })

                return json.dumps({
                    "status": "found",
                    "candidates": _candidates_found,
                    "venue": venue_name,
                    "image": str(filepath)
                })

            time.sleep(0.3)

        except Exception as e:
            print(f"      ❌ Error: {e}")
            continue

    # If no Street View angle worked, try Google Places photos
    print("   No suitable Street View images. Trying Google Places photos...")
    return try_google_places_photos(venue_name, address, rating)


def validate_storefront_image(image_b64: str, venue_name: str, venue_address: str) -> dict:
    """Validate a storefront image with GPT-4V"""
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""You are an expert evaluating storefronts for planter installation visualization.
Analyzing: {venue_name} at {venue_address}

STRICT criteria - ALL must be true:
1. Can you see THIS VENUE'S signage, name, or branding? (NOT just "a storefront" on the street)
2. Is it unambiguously THIS VENUE'S storefront (not a neighboring shop)?
3. Is the actual front door/entrance clearly visible?
4. Is the FULL storefront view complete and unobstructed? (not cut off, not partial)
5. Is the storefront bare/plain enough that planters would visibly improve it?
6. Would planters realistically fit and look good here for a before/after visualization?

CRITICAL: If you cannot definitively identify this as {venue_name}'s storefront, OR if the view is partial/cut off, return suitable=false. The image must show a COMPLETE storefront view suitable for compositing planter visualizations.

Return JSON:
{{
  "is_storefront": true/false,
  "entrance_visible": true/false,
  "full_view": true/false,
  "bare_enough": true/false,
  "venue_identified": true/false,
  "suitable": true/false,
  "confidence": <0-100>,
  "reason": "<brief reason>"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ]
        )

        response_text = response.choices[0].message.content.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        return json.loads(response_text)
    except Exception as e:
        return {
            "is_storefront": False,
            "entrance_visible": False,
            "bare_enough": False,
            "suitable": False,
            "confidence": 0,
            "reason": f"Error: {str(e)}"
        }


def try_google_places_photos(venue_name: str, address: str, rating: Optional[float]) -> str:
    """Try to find a suitable image from Google Places photos"""
    global _candidates_found, _venues_with_images

    if venue_name not in _venue_photos or not _venue_photos[venue_name]:
        print("   ⚠️  No Google Places photos available")
        return json.dumps({
            "status": "no_suitable_image",
            "venue": venue_name
        })

    print("   📸 Trying Google Places photos...")
    photos_meta = _venue_photos[venue_name]
    print(f"   Found {len(photos_meta)} Google Places photos to check")

    for i, photo in enumerate(photos_meta[:30]):  # Try up to 30 photos
        photo_name = photo.get("name")
        if not photo_name:
            continue

        try:
            # Get photo URL
            url = f"https://places.googleapis.com/v1/{photo_name}/media"
            params = {"key": MAPS_API_KEY, "maxWidthPx": 600}
            url_resp = requests.get(url, params=params, allow_redirects=False, timeout=10)

            if url_resp.status_code == 302 and "Location" in url_resp.headers:
                photo_url = url_resp.headers["Location"]
            elif url_resp.ok:
                photo_url = url_resp.url
            else:
                continue

            # Download photo
            photo_resp = requests.get(photo_url, timeout=10)
            if not photo_resp.ok:
                continue

            image_b64 = base64.b64encode(photo_resp.content).decode()

            # Validate image
            print(f"      🔍 Validating Google Places photo {i+1}...")
            validation = validate_storefront_image(image_b64, venue_name, address)

            is_suitable = (
                validation.get("suitable", False) and
                validation.get("confidence", 0) >= 70
            )

            print(f"      Storefront: {validation.get('is_storefront')}, "
                  f"Visible: {validation.get('entrance_visible')}, "
                  f"Bare: {validation.get('bare_enough')}, "
                  f"Suitable: {validation.get('suitable')}, "
                  f"Confidence: {validation.get('confidence')}%")

            # ALWAYS save analyzed images for debugging (organized by venue)
            venue_dir = GOOGLEPLACES_DIR / venue_name.replace(' ', '_')
            venue_dir.mkdir(parents=True, exist_ok=True)
            filename = f"photo_{i+1}.jpg"
            filepath = venue_dir / filename
            filepath.write_bytes(photo_resp.content)
            print(f"      💾 Saved to google_places/{venue_name.replace(' ', '_')}/{filename}")

            # Log this analyzed image
            _all_analyzed_images.append({
                "type": "google_places",
                "venue": venue_name,
                "address": address,
                "file": str(filepath),
                "photo_index": i+1,
                "validation": validation,
                "is_suitable": is_suitable,
                "reason_rejected": None if is_suitable else f"Score {validation.get('confidence')}% < 70% or suitable={validation.get('suitable')}"
            })

            if is_suitable:
                # Check if this is a new venue
                is_new_venue = venue_name not in _venues_with_images

                if is_new_venue:
                    _candidates_found += 1
                    _venues_with_images.add(venue_name)
                    # Create candidates folder on first suitable image
                    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

                # Also save to candidates folder
                candidates_venue_dir = CANDIDATES_DIR / venue_name.replace(' ', '_')
                candidates_venue_dir.mkdir(parents=True, exist_ok=True)
                candidates_filepath = candidates_venue_dir / filename
                candidates_filepath.write_bytes(photo_resp.content)

                print(f"      ✅ ACCEPTED! Image saved: {filename}")

                _analyzed_venues.append({
                    "name": venue_name,
                    "address": address,
                    "rating": rating,
                    "image": str(filepath),
                    "heading": f"Google Places photo {i+1}",
                    "validation": validation
                })

                if _candidates_found >= 3:
                    return json.dumps({
                        "status": "success",
                        "candidates": _candidates_found,
                        "venue": venue_name
                    })

                return json.dumps({
                    "status": "found",
                    "candidates": _candidates_found,
                    "venue": venue_name
                })

            time.sleep(0.2)

        except Exception as e:
            print(f"      ❌ Error: {e}")
            continue

    return json.dumps({
        "status": "no_suitable_image",
        "venue": venue_name
    })


@function_tool
def get_candidates_count() -> str:
    """Get current count of candidates found"""
    return json.dumps({
        "count": _candidates_found,
        "venues": [{"name": v["name"], "address": v["address"], "image": v["image"]}
                   for v in _analyzed_venues]
    })


@function_tool
def get_final_candidates() -> str:
    """Get the final 3 candidates - only call once you have 3"""
    if _candidates_found < 3:
        return json.dumps({
            "status": "incomplete",
            "count": _candidates_found,
            "message": f"Only {_candidates_found} candidates found, need 3"
        })

    return json.dumps({
        "status": "complete",
        "count": _candidates_found,
        "candidates": _analyzed_venues[:3]
    })



async def main():
    """Run the autonomous agent"""
    agent = Agent(
        name="Storefront Discovery Agent",
        instructions="""Find exactly 3 storefronts in London suitable for planters, then STOP.

WORKFLOW:
1. Search DIFFERENT London neighborhoods (Soho, Shoreditch, Camden, Covent Garden, Fitzrovia, Clerkenwell, Brick Lane, Notting Hill, Chelsea, Carnaby Street)
2. For each neighborhood, search for cafes/restaurants/salons/boutiques/bakeries
3. For each venue with Street View, call analyze_venue_candidate (it handles 8 angles + fallback)
4. IMPORTANT: When analyze_venue_candidate returns {"status": "halt"}, STOP ALL SEARCHING immediately
5. Once you receive halt, call get_final_candidates and return that as your final answer

CRITICAL RULES:
- Search DIFFERENT neighborhoods to get diverse candidates (not the same venues repeatedly)
- Do NOT search for more venues once any result contains "halt"
- Do NOT call analyze_venue_candidate after receiving halt status
- Must find exactly 3 candidate venues (not 3 images from fewer venues)
- Once halt is received, only call get_final_candidates
- Stop immediately - do not continue searching""",
        tools=[
            search_venues,
            check_street_view,
            analyze_venue_candidate,
            get_candidates_count,
            get_final_candidates,
        ],
    )

    print("\n" + "🤖 " * 20)
    print("AUTONOMOUS STOREFRONT DISCOVERY AGENT")
    print("🤖 " * 20 + "\n")

    result = await Runner.run(
        agent,
        "Find 3 storefronts in London for planters. Search venues, analyze each with Street View (8 angles), save images. Stop at 3 candidates.",
        max_turns=50,
    )

    print("\n" + "="*70)
    print("AGENT COMPLETED")
    print("="*70)
    print(f"\nAgent output:\n{result.final_output}\n")

    # Save final results with comprehensive analysis
    output_file = RUN_DIR / "results.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "run_id": RUN_TIMESTAMP,
            "summary": {
                "total_images_analyzed": len(_all_analyzed_images),
                "candidates_found": _candidates_found,
                "suitable_count": len(_analyzed_venues),
            },
            "candidates": _analyzed_venues,
            "all_analyzed_images": _all_analyzed_images
        }, f, indent=2)

    print(f"\n💾 Run saved to: {RUN_DIR}")
    print(f"📸 360 View images: {STREETVIEW_DIR}")
    print(f"📸 Google Places images: {GOOGLEPLACES_DIR}")

    if _analyzed_venues:
        print(f"🏆 Candidates folder: {CANDIDATES_DIR}")
        print("\n📊 FINAL CANDIDATES:\n")
        for i, venue in enumerate(_analyzed_venues, 1):
            print(f"{i}. {venue['name']}")
            print(f"   Address: {venue['address']}")
            print(f"   Rating: {venue['rating']}")
            print(f"   Image: {Path(venue['image']).name}")
            print(f"   Source: {venue.get('heading', 'Unknown')}")
            print(f"   Validation: {venue['validation']}")
            print()
    else:
        print("\n⚠️  NO SUITABLE CANDIDATES FOUND")
        print("   Check analyzed images in 360_view/ and google_places/ folders")
        print("   Review the validation scores to see why they were rejected")


if __name__ == "__main__":
    asyncio.run(main())
