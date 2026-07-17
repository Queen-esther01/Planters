# Storefront Planter Compositing System - Design Document

## Executive Summary

This document covers the design and implementation of an automated system that identifies suitable storefronts in London and composites design-led outdoor planters onto their entrances to produce realistic "what it could look like" visualizations for prospecting purposes.

The system has two core components:
1. **Storefront Discovery & Capture** — Finding candidate venues and retrieving well-framed frontage images
2. **Planter Compositing** — Extracting planter product photography and placing it onto storefronts at correct scale and perspective



## Installation & Setup

### Dependencies

Install required Python packages:

```bash
pip install opencv-python numpy pathlib ultralytics google-cloud-places requests python-dotenv openai streamlit pillow
```

Or using UV (faster):
```bash
uv add opencv-python numpy pathlib ultralytics google-cloud-places requests python-dotenv openai streamlit pillow
```

### Environment Variables & API Keys

Create a `.env` file in the project root with your API keys:

```
MAPS_API_KEY=your_google_maps_api_key
OPENAI_API_KEY=your_openai_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

**Required API Keys:**

1. **Google Maps API Key** (`MAPS_API_KEY`)
   - **Purpose:** Street View API, Places API, Geocoding
   - **Get it:** [Google Cloud Console](https://console.cloud.google.com/)
   - **Setup:**
     1. Create a new GCP project
     2. Enable APIs: Street View Static API, Places API (New)
     3. Create an API key (Credentials > Create Credentials > API Key)
   - **Cost:** Street View & Places API have free tier; excess usage charged
   - **Used by:** `autonomous_agent.py` for venue discovery and Street View imagery

2. **OpenAI API Key** (`OPENAI_API_KEY`)
   - **Purpose:** GPT-4 Vision for storefront evaluation and compositing
   - **Get it:** [OpenAI Platform](https://platform.openai.com/account/api-keys)
   - **Setup:**
     1. Create an OpenAI account
     2. Go to API keys section
     3. Create a new secret key
   - **Cost:** Pay-as-you-go; ~$0.01-0.05 per storefront evaluation
   - **Used by:** `autonomous_agent.py` for bareness/visibility scoring

3. **DeepSeek API Key** (`DEEPSEEK_API_KEY`) — *Optional, not currently used*
   - **Purpose:** Alternative vision model (evaluated but abandoned due to reliability issues)
   - **Note:** Can be omitted; system defaults to OpenAI if not set
   - **Used by:** (None - kept for reference)

---

## Running the System

### Frontend (Web UI)

Start the Streamlit web application:

```bash
streamlit run app_streamlit.py
```

Opens at: `http://localhost:8501`

**Features:**
- **Tab 1: Discover Candidates** — Run autonomous agent to find storefronts
- **Tab 2: Composite Planters** — Select a venue and composite planters onto it
- **Sidebar:** Choose between Plant 1 or Plant 3

### Command Line (Standalone)
Note: The CLI is primarily for testing and debugging so you can view logs of agent activity; the Streamlit UI is the main interface, there are no logs.
#### 1. Extract Planter Cutout (Preview)

Visualize SAM+YOLO segmentation on a planter product photo:

```bash
python planter_compositor.py --preview "planters/plant_1.png" -o planter_preview.png
```

Output: `planter_preview.png` — planter on transparent background

### 2. Visualize Storefront Detections

See what YOLO detects in a storefront image:
```bash
python planter_compositor.py --visualize "agent_output/20260717_120646/candidates/Luna_Curious/photo_1.jpg"
```

Outputs:
- `detection_full.jpg` — Full image with all detections labeled
- `detection_door.jpg` — Cropped door region
- `detection_ground.jpg` — Cropped ground region

### 3. Composite Planters onto Storefront

Generate the final "what it could look like" visualization:

```bash
python planter_compositor.py "agent_output/20260717_123434/candidates/Kricket_Shoreditch/photo_9.jpg" "planters/plant_1.png" -o composited_result.jpg
```

Supports multiple planters:
```bash
python planter_compositor.py "storefront.jpg" "planter1.png" "planter2.png" -o result.jpg
```

Output: `composited_result.jpg` — composited image with planters positioned on storefront

---

## Technical Stack
Note: All required models will be downloaded automatically on first run.
**Dependencies:**
- `ultralytics` — YOLO26 for bounding boxes & SAM2 segmentation
- `opencv-python` — Image processing (resize, mask operations, compositing)
- `google-cloud-places` / `requests` — Google Places API for venue discovery
- `openai` — GPT-4 Vision for storefront evaluation

**Models Used:**
- **SAM2 (`sam2_l.pt`)** — Large model for precise plant segmentation
- **YOLO26 (`yolov26m-seg.pt`)** — Medium segmentation model for object detection & door detection

**Data:**
- Input: Planter product photos (`planters/plant_*.png`)
- Input: Storefront Street View imagery from Google Maps API
- Output: Composited visualizations + detection previews

---

## Known Limitations & Future Work

### Current Limitations

1. **No Perspective Correction** — Planters are pasted at original scale, not corrected for perspective depth
2. **No Shadow Generation** — Planters lack realistic shadows; appear to float
3. **No Lighting Integration** — No color/brightness correction to match scene lighting
4. **Door Detection Unreliable** — Falls back to hardcoded positioning
5. **Single Positioning** — Planters always bottom-left/right; no flexible placement
6. **No Ground Validation** — Composited planters not checked to be on ground/surface

### Priority for Next Phase

1. **Fix Door Detection** — Use a specialized model or better heuristics (edges, lines, symmetry)
2. **Implement Perspective Scaling** — Use detected door height to scale planters correctly
3. **Add Shadow Generation** — Infer lighting direction from image, generate shadows under planters
4. **Lighting Matching** — Adjust planter brightness/color to match storefront scene
5. **Rejection Filtering** — Automated checks for unrealistic composites before output

---


