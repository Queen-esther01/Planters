# Storefront Planter Compositing System - Design Document

## Executive Summary

This document covers the design and implementation of an automated system that identifies suitable storefronts in London and composites design-led outdoor planters onto their entrances to produce realistic "what it could look like" visualizations for prospecting purposes.

The system has two core components:
1. **Storefront Discovery & Capture** — Finding candidate venues and retrieving well-framed frontage images
2. **Planter Compositing** — Extracting planter product photography and placing it onto storefronts at correct scale and perspective

## Part 1: Storefront Discovery & Capture

### Approach

The discovery pipeline uses:
1. **Google Places API** — Search for independent cafes, restaurants, salons, and similar street-facing businesses in London
2. **Street View API** — Retrieve 8-directional imagery (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°) for each venue to cover cases where the entrance is not directly facing the street
3. **GPT-4 Vision** — Evaluate frontage "bareness" (1-10 scale) and visibility to identify suitable candidates
4. **Automated filtering** — Only venues with high-confidence bare storefronts (score ≥8, visible=YES) are accepted

### Rejection Criteria for Candidates

A venue is rejected if:
- No Street View coverage available at coordinates
- Storefront not clearly visible (interior shots, wrong angle, obstructed)
- Bareness score < 8 (already has significant decoration, planters, or visual clutter)
- Business type doesn't match target categories (must be independent street-facing retail/service)

### Rejection Criteria for Framing

An image framing is rejected if:
- The entrance/doorway is not in the foreground (>5m away or blocked by other objects)
- Image faces wrong direction (side alley, wrong side of building)
- Dramatic shadows or poor lighting obscure architectural details
- Fallback: If Street View fails, attempt Google Business photos; if those fail, skip the venue

### Imagery Fallback Strategy

When Street View capture fails or produces poor framing:

1. **Primary:** Google Street View API with 8 directional angles (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
   - Covers 360° to find optimal entrance framing
   - Automated framing selection based on visibility score

2. **Secondary:** Google Places API Business Photos
   - Retrieved if Street View produces no suitable images (visibility < threshold)
   - Uses venue's own uploaded photos (often shows best presentation)
   - Fallback limit: try up to 30 available photos before rejecting venue

3. **Tertiary:** Skip venue if both sources fail to produce usable frontage

### Data Storage & Results

**Directory Structure:**
```
agent_output/
└── {timestamp}/
    ├── 360_view/
    │   └── {venue_name}/
    │       ├── N.jpg (0°)
    │       ├── NE.jpg (45°)
    │       ├── E.jpg (90°)
    │       ├── SE.jpg (135°)
    │       ├── S.jpg (180°)
    │       ├── SW.jpg (225°)
    │       ├── W.jpg (270°)
    │       └── NW.jpg (315°)
    └── results.json
```

**results.json Schema:**

Contains structured selection/rejection data for all evaluated venues:

```json
{
  "candidates": [
    {
      "name": "Collectif - Stables Market (Camden)",
      "address": "Stables Market, Camden, London",
      "coordinates": {"lat": 51.5407, "lng": -0.0946},
      "business_type": "cafe",
      "selected": true,
      "selection_reason": "Bare storefront, clear visibility, high bareness score",
      "street_view_coverage": true,
      "photos_retrieved": 8,
      "best_angle": "N (0°)",
      "bareness_scores": [
        {"angle": "N", "score": 9, "visible": "YES", "reason": "..."},
        {"angle": "NE", "score": 7, "visible": "YES", "reason": "..."}
      ]
    },
    {
      "name": "Rejected Venue Example",
      "address": "...",
      "coordinates": {"lat": 51.5xxx, "lng": -0.1xxx},
      "selected": false,
      "rejection_reason": "Bareness score < 8 (already decorated with planters)",
      "street_view_coverage": true,
      "best_score": 4
    }
  ],
  "summary": {
    "total_places_searched": 280,
    "with_street_view": 279,
    "candidates_selected": 3,
    "rejection_breakdown": {
      "low_bareness_score": 156,
      "poor_visibility": 54,
      "no_street_view": 1,
      "business_type_mismatch": 65
    }
  }
}
```

**Key Fields in results.json:**
- `selection_reason` / `rejection_reason` — Human-readable explanation of decision
- `bareness_scores` — Per-angle evaluation results (score 1-10, visibility YES/NO, GPT reasoning)
- `best_angle` — Which directional photo was best for final selection
- `photos_retrieved` — Number of 8-directional Street View captures attempted
- `rejection_breakdown` — Aggregate stats on why candidates were filtered out

### Current State

The autonomous agent (`autonomous_agent.py`) successfully:
- Discovers 280+ candidate venues across London using Google Places API
- Checks Street View coverage (279/280 venues have imagery available)
- Captures 8-directional Street View imagery for each venue (up to 8 angles per location)
- Falls back to Google Places Business Photos if Street View is unavailable
- Evaluates each angle with GPT-4 Vision for bareness (1-10 scale) and visibility (YES/NO)
- Selects venues meeting criteria (bareness ≥8, visible=YES, independent street-facing business)
- Saves all results and metadata to `agent_output/{timestamp}/results.json`

**Example Selected candidate venues (3 total):**

1. **Jolene Redchurch Street** — 67 Redchurch St, London E2 7DJ
   - Image source: Google Places photo 3
   - Confidence: 90%
   - Assessment: Complete and unobstructed view of storefront with signage, entrance, and space for planters

3. **The House of Retro** — 232 Portobello Rd, London W11 1LJ
   - Image source: Google Places photo 1
   - Confidence: 100%
   - Assessment: Clear complete view of storefront with visible signage and entrance. Plain enough to benefit from planter enhancement

4. **Appletree Boutique** — 47 Pembridge Rd, Notting Hill Gate, London W11 3HG
   - Image source: Google Places photo 5
   - Confidence: 95%
   - Assessment: Complete unobstructed view with visible signage. Plain storefront suitable for planter installation

**Discovery Insights:**

- Total images analyzed: 74 (8 street view angles per venue × selected venues + Google Places photos)
- Street View coverage: Most Street View angles showed interior views or poor framing (rejected as unsuitable)
- Fallback effectiveness: Google Places photos provided superior framing — all 4 selected candidates came from business photos, not Street View. However, it is pertinent to note that the Google places photos are not always the best representation of the storefront, as some were taken from inside the venue or at odd angles. Case in point: The Appletree Boutique image was taken at an odd angle & leaves very little space for planters. The agent prompt will need to be tightened.
- Selection threshold: All candidates met confidence ≥90% and had complete, unobstructed views with visible signage
- Rejection breakdown: 70 images rejected for: interior views (32), obstructed framing (18), partial views (12), misidentified storefronts (8)

### Vision Model & API Choices

**Vision Evaluation:** Originally attempted DeepSeek for storefront evaluation but encountered usability issues. Switched to **OpenAI GPT-4 Vision** for:
- Consistent multi-image evaluation across candidate venues
- Reliable JSON response parsing for scoring
- Better handling of ambiguous storefronts

### Filtering Challenges Identified

During development, several filtering issues emerged that required stricter rejection criteria:

1. **Business Name Mismatch** — Some Street View angles don't show business signage clearly; system can't confirm it's the correct storefront
2. **Wrong Storefront Detection** — System sometimes selected adjacent storefronts or landmarks instead of the target business's entrance
3. **Storefront Visibility** — Images showed storefronts but from wrong angles, too far away, or blocked by other objects
4. **Solution Implemented** — Let go of ambiguous images; enforce "clearly visible entrance in foreground" requirement

**Recommendation for Future:** Fine-tune YOLO on storefront detection rather than relying on generic object detection. Custom model trained on storefront entrance geometry (door height, window frame patterns, fascia) would improve:
- Consistent door/entrance localization
- Rejection of non-storefront elements (parked cars, signage, side walls)
- Scale estimation accuracy from detected architectural features

---

## Part 2: Planter Compositing

### Segmentation Strategy

**The Challenge:** Extract product planters from reference photography cleanly, preserving fine details without artifacts.

**Evolution of Approaches:**

1. **Poisson Blending (REJECTED)** ❌
   - Method: Use YOLO to detect planter, apply cv2.seamlessClone()
   - Result: Planters lacked shadows, grounding, and looked artificially pasted
   - Problem: No shadow generation or lighting integration
   - Feedback: "The poisson blending is terrible!"

2. **OpenAI Image Edit API (REJECTED)** ❌
   - Method: Use generative inpainting with YOLO bounding box as mask
   - Result: Still looked "terrible" even with generative approach
   - Problem: Hallucinated background, inconsistent lighting
   - Feedback: "looks terrible"

3. **SAM Standalone Full-Image Segmentation (REJECTED)** ❌
   - Method: Run SAM2 on entire image without guidance
   - Result: Detected 14 objects in grouped planter images, but segmentation was imprecise
   - Problem: Over-segmented individual pots; couldn't handle grouped planters well
   - User observation: "sam seems to only segment the small box even though the planters are kind of joined together"

4. **YOLO + SAM (ACCEPTED)** ✅
   - Method: YOLO detects planter (class awareness), returns bounding box → SAM segments within that box (precision)
   - Result: Clean, precise masks with fine detail preservation
   - Feedback: "The segmentation sam file being produced looks really good"
   - Output: `segmentation_sam.png` with high-quality cutout

### Final Segmentation Implementation

**Method: YOLO-Prompted SAM2**

```
Input: Planter product photo
├─ Step 1: Run YOLO26 to detect objects
│  └─ Search for object class containing "potted plant" or "plant"
│  └─ Extract bounding box of detected plant
├─ Step 2: Use bounding box to prompt SAM2
│  └─ SAM2 segments within that region with high precision
│  └─ Returns clean mask of plant pixels
├─ Step 3: Create RGBA image
│  └─ Alpha channel = SAM mask (as-is, no refinements)
│  └─ RGB channels = original image
└─ Output: RGBA planter cutout on transparent background
```

**Key Design Decisions:**

1. **No Mask Refinements** — SAM's mask is already precise. Morphological operations (erosion, dilation) and Gaussian blur degrade the quality.
2. **Direct Mask Usage** — Use SAM's output directly as alpha; don't over-process.
3. **Bounding Box Prompting** — YOLO's class awareness + SAM's precision is the optimal combination.

### Compositing Strategy

**The Challenge:** Place extracted planters onto storefront images believably, at correct scale and perspective.

**Evolution:**

1. **Multi-Band Pyramid Blending (REJECTED)** ❌
   - Attempted sophisticated frequency-domain blending with Gaussian pyramids
   - Result: Shape mismatches during upsampling, caused crashes
   - Problem: Over-engineered for current requirements
   - Removed: Unnecessary complexity

2. **Alpha Blending with Refinement (REJECTED)** ❌
   - Attempted alpha blending with feathering and edge smoothing
   - Result: Made cutouts look semi-transparent and unnatural
   - Feedback: "remove the blending bullshit"

3. **Hard Paste Compositing (ACCEPTED)** ✅
   - Method: Paste planter pixels directly where alpha > 0
   - Result: Maintains crisp cutout edges, no transparency artifacts
   - Current approach: Simple, effective, fast

**Current Compositing Pipeline:**

```
Input: Storefront image + Planter RGBA cutout
├─ Load storefront image
├─ Extract planter with YOLO+SAM (as-is, no refinements)
├─ Determine planter size (no scaling yet)
├─ Position at bottom-left or bottom-right (hard-coded x positions)
├─ Clip planter to fit within storefront bounds
│  └─ If planter taller than remaining space, crop bottom portion
│  └─ If planter extends beyond image edges, crop sides
│  └─ Skip planter if crop results in zero area
├─ Hard-paste: result[y:y+crop_h, x:x+crop_w][alpha>0] = planter_rgb[alpha>0]
└─ Output: Composited image with cropped/fitted planter
```

**Bounds Clipping Logic:**

- Calculate crop region: `crop_x, crop_y, crop_w, crop_h`
- Clip position `(x, y)` to valid range within storefront
- Crop foreground planter to intersection with storefront bounds
- Only paste visible region where alpha channel > 0
- Skip planter if final crop area is zero (doesn't fit at all)

**What's Deferred (Future Work):**

- Perspective-aware scaling based on door/entrance height
- Realistic shadow generation (direction, softness based on lighting)
- Lighting integration (color correction, brightness matching)
- Multiple planter positioning (left/right of door)

**Rejection Criteria for Bad Composites:**

A composite should be rejected if:
- Planter appears to float (no ground contact)
- Severe lighting mismatch (planter shadow direction opposite to scene)
- Scale wildly incorrect (planter 10x larger/smaller than realistic)
- Halos or rough blending edges around cutout
- Planter obscures critical storefront features (entrance, signage)

---

## Scale Estimation (Future Implementation)

**Approach:** Use reference objects to derive real-world scale.

Door/entrance dimensions can be estimated from:
1. **Typical UK door height:** ~2.1m (standard internal), ~2.4m (commercial)
2. **Window panes:** Standard sash windows ~1.5m tall
3. **Street furniture:** Bollards ~1.2m, postboxes ~1.5m

When a door is detected, measure its pixel height in the image and derive:
```
pixels_per_meter = door_pixel_height / door_real_height_meters
```

Then scale planters: `planter_display_height = planter_real_height * pixels_per_meter`

**Current Status:** Door detection partially implemented. The door detection is meant to be used to decide best place to place planters. YOLO often fails to detect actual doors (confuses them with people or windows). Fallback logic estimates door dimensions from image center-bottom.

---

## Door Detection Issues & Current Workarounds

### Problem Identified

YOLO's generic detection model doesn't reliably identify storefront doors:
- Misses storefronts entirely in most framings
- Detects windows/fascia instead of entrance

### Current Solution

When door detection fails:
1. Fall back to hardcoded door estimation: center-bottom of image, 20% width × 40% height
2. Position planters at bottom-left/bottom-right edges (fixed offset)
3. No perspective correction applied yet

### Detection Visualization

Use the `--visualize` flag to inspect what YOLO is detecting:
```bash
python planter_compositor.py --visualize "agent_output/20260717_120646/candidates/Luna_Curious/photo_1.jpg"
```

Outputs:
- `detection_full.jpg` — All detections with bounding boxes
- `detection_door.jpg` — Cropped region of detected door (or empty if none found)
- `detection_ground.jpg` — Cropped region of detected ground (or empty if none found)

---

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

## Design Rationale

### Why YOLO + SAM (Not Just SAM)?

SAM is excellent at precise segmentation *within context*, but without a prompt (bounding box or point), it over-segments grouped objects. YOLO provides:
- Class awareness ("this is a plant")
- Bounding box localization
- Fallback to largest object if no plant detected

SAM then refines within that context, giving both accuracy and speed.

### Why Hard Paste (Not Blending)?

Alpha blending with edge feathering makes cutouts look semi-transparent and unnatural. Hard-pasting (only where alpha > 0) preserves the sharpness of the SAM mask and looks more like a real object placed into the scene. Future versions can add sophisticated blending *after* perspective/shadow work, but for initial validation, hard paste is correct.

### Why Street View Over Generic Stock?

Real storefront imagery is required to:
- Show the *actual* entrance customers see
- Maintain architectural authenticity for outreach credibility
- Enable accurate scale/perspective from real proportions
- Build trust with venue owners (this is YOUR storefront, not a simulation)

### Imagery Rights Position

**Position:** Capture and reuse of Street View imagery for commercial prospecting raises legal and ethical concerns:

- Street View imagery is ©Google; commercial reuse requires permission or paid license
- Using real storefronts without owner consent to solicit business is legally uncertain (privacy/publicity rights)
- Realistic composites showing "planters installed" could be perceived as misleading if used before explicit business discussion

**Recommendation for Production:**
1. Obtain explicit permission from venue before generating/sending composites
2. Use composites only as prospecting aids (shared after permission), not mass-marketing assets
3. Consider licensing Street View imagery from Google for commercial use
4. Frame composites as "mock-ups for discussion purposes only"

---

## Deliverables Summary

1. **`autonomous_agent.py`** — Discovers venues, evaluates storefronts, retrieves Street View imagery
2. **`planter_compositor.py`** — Segments planters (YOLO+SAM) and composites onto storefronts
3. **`app.ipynb`** — Jupyter notebook with segmentation experiments and comparisons
4. **Example Outputs:**
   - `detection_full.jpg` — YOLO detections
   - `planter_preview.png` — Extracted planter cutout
   - `composited_result.jpg` — Final "what it could look like" visualization

---

## Next Steps (Future Phases)

**Phase 2 (Realism):**
- [ ] Implement perspective-aware scaling
- [ ] Generate realistic shadows
- [ ] Add lighting/color correction
- [ ] Deploy compositing to web interface

