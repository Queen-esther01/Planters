import streamlit as st
import json
import os
from pathlib import Path
from datetime import datetime
import subprocess
import cv2
from PIL import Image
import time

st.set_page_config(page_title="Planter Compositor", layout="wide")

BASE_DIR = Path(os.environ.get("PLANTER_BASE_DIR", Path(__file__).resolve().parent))

st.title("🪴 Storefront Planter Compositor")
st.markdown("Generate realistic planter visualizations for London storefronts")

# Sidebar for configuration
st.sidebar.header("Configuration")
planter_choice = st.sidebar.radio("Select Planter Product", ["Plant 1", "Plant 3"])
planter_map = {
    "Plant 1": BASE_DIR / "planters/plant_1.png",
    "Plant 3": BASE_DIR / "planters/plant_3.png",
}
planter_path = planter_map[planter_choice]

# Main tabs
tab1, tab2 = st.tabs(["📍 Discover Candidates", "🎨 Composite Planters"])

# ============================================================
# TAB 1: DISCOVER CANDIDATES
# ============================================================
with tab1:
    st.header("Step 1: Discover Candidate Venues")
    st.markdown("""
    This tool identifies bare storefronts in London that would benefit from planter installations.
    - Uses Google Places API to find venues
    - Evaluates with GPT-4 Vision for bareness and visibility
    - Selects only high-quality candidates
    """)

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("🔍 Generate Candidates", key="gen_candidates", use_container_width=True):
            st.info("Running autonomous agent discovery... (this may take 2-3 minutes)")

            # Run the autonomous agent
            try:
                result = subprocess.run(
                    ["python", "autonomous_agent.py"],
                    cwd=str(BASE_DIR),
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                if result.returncode == 0:
                    st.success("✅ Agent completed successfully!")
                    st.balloons()
                else:
                    st.error(f"❌ Agent failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                st.error("❌ Agent timeout (>5 minutes)")
            except Exception as e:
                st.error(f"❌ Error running agent: {str(e)}")

    # Load and display candidates
    results_pattern = BASE_DIR / "agent_output"
    latest_run = None

    if results_pattern.exists():
        # Find latest run directory
        run_dirs = sorted(results_pattern.glob("*"), key=lambda x: x.name, reverse=True)
        if run_dirs:
            latest_run = run_dirs[0]
            results_file = latest_run / "results.json"

            if results_file.exists():
                st.success(f"✅ Loaded results from {latest_run.name}")

                with open(results_file, "r") as f:
                    results = json.load(f)

                # Display summary
                summary = results.get("summary", {})
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Analyzed", summary.get("total_images_analyzed", 0))
                col2.metric("Candidates Found", summary.get("candidates_found", 0))
                col3.metric("Suitable Count", summary.get("suitable_count", 0))
                col4.metric("Run ID", latest_run.name)

                # Display candidates
                st.subheader("Selected Candidates")
                candidates = results.get("candidates", [])

                if candidates:
                    for idx, candidate in enumerate(candidates, 1):
                        with st.expander(
                            f"#{idx} {candidate['name']} - {candidate.get('rating', 'N/A')}⭐",
                            expanded=(idx == 1)
                        ):
                            col1, col2 = st.columns([1, 1])

                            with col1:
                                st.markdown("**Venue Details**")
                                st.text(f"Name: {candidate['name']}")
                                st.text(f"Address: {candidate['address']}")
                                st.text(f"Rating: {candidate.get('rating', 'N/A')}⭐")

                                validation = candidate.get("validation", {})
                                st.markdown("**Validation**")
                                st.text(f"Is Storefront: {validation.get('is_storefront', False)}")
                                st.text(f"Entrance Visible: {validation.get('entrance_visible', False)}")
                                st.text(f"Full View: {validation.get('full_view', False)}")
                                st.text(f"Bare Enough: {validation.get('bare_enough', False)}")
                                st.text(f"Confidence: {validation.get('confidence', 0)}%")

                            with col2:
                                st.markdown("**Selection Reason**")
                                reason = validation.get("reason", "No reason provided")
                                st.info(reason)

                                # Display image if available
                                image_path = candidate.get("image", "")
                                if image_path:
                                    full_path = BASE_DIR / image_path
                                    if full_path.exists():
                                        st.markdown("**Storefront Image**")
                                        st.image(str(full_path), use_column_width=True)
                else:
                    st.warning("No candidates found in results")
            else:
                st.warning("No results.json found in latest run")
    else:
        st.info("👈 Run the agent to discover candidates")

# ============================================================
# TAB 2: COMPOSITE PLANTERS
# ============================================================
with tab2:
    st.header("Step 2: Composite Planters onto Storefronts")
    st.markdown(f"""
    Using: **{planter_choice}** ({planter_path})
    """)

    # Load candidates for selection
    results_pattern = BASE_DIR / "agent_output"
    latest_run = None
    candidates = []

    if results_pattern.exists():
        run_dirs = sorted(results_pattern.glob("*"), key=lambda x: x.name, reverse=True)
        if run_dirs:
            latest_run = run_dirs[0]
            results_file = latest_run / "results.json"

            if results_file.exists():
                with open(results_file, "r") as f:
                    results = json.load(f)
                    candidates = results.get("candidates", [])

    if not candidates:
        st.warning("No candidates available. Run discovery first in the 'Discover Candidates' tab.")
    else:
        # Create a mapping of candidate names to image paths
        candidate_options = {f"{c['name']} ({c['address']})": c for c in candidates}

        selected_candidate_name = st.selectbox(
            "Select a storefront to composite:",
            list(candidate_options.keys()),
            index=1
        )

        if selected_candidate_name:
            selected = candidate_options[selected_candidate_name]

            # Show selected venue info
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Selected Venue**")
                st.text(f"Name: {selected['name']}")
                st.text(f"Address: {selected['address']}")

            with col2:
                st.markdown("**Image Source**")
                st.text(f"Type: {selected.get('heading', 'N/A')}")
                confidence = selected.get('validation', {}).get('confidence', 0)
                st.text(f"Confidence: {confidence}%")

            # Show preview images side by side
            image_path = selected.get("image", "")
            if image_path:
                full_path = BASE_DIR / image_path
                if full_path.exists():
                    st.markdown("**Preview**")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Planter Product**")
                        if planter_path.exists():
                            st.image(str(planter_path), use_column_width=True)
                        else:
                            st.warning(f"Planter image not found: {planter_path}")

                    with col2:
                        st.markdown("**Target Storefront**")
                        st.image(str(full_path), use_column_width=True)

            # Generate composite button
            st.divider()
            if st.button("🎨 Generate Composite", use_container_width=True, type="primary"):
                st.info("Generating composite image... (this may take 1-2 minutes)")

                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Run the compositor
                    status_text.text("Loading models...")
                    progress_bar.progress(25)

                    storefront_path = image_path
                    output_path = "composited_result.jpg"

                    result = subprocess.run(
                        [
                            "python",
                            "planter_compositor.py",
                            storefront_path,
                            str(planter_path),
                            "-o",
                            output_path
                        ],
                        cwd=str(BASE_DIR),
                        capture_output=True,
                        text=True,
                        timeout=180
                    )

                    progress_bar.progress(75)
                    status_text.text("Processing complete!")

                    if result.returncode == 0:
                        progress_bar.progress(100)
                        st.success("✅ Composite generated successfully!")

                        # Display result side by side
                        output_full_path = BASE_DIR / output_path
                        if output_full_path.exists():
                            st.markdown("### 🎉 Comparison")
                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("**Original Storefront**")
                                original_img = Image.open(full_path)
                                st.image(original_img, use_column_width=True)

                            with col2:
                                st.markdown("**With Planters**")
                                result_img = Image.open(output_full_path)
                                st.image(result_img, use_column_width=True)

                            # Provide download
                            st.divider()
                            with open(output_full_path, "rb") as f:
                                st.download_button(
                                    label="⬇️ Download Composite Image",
                                    data=f.read(),
                                    file_name=f"planter_composite_{selected['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                                    mime="image/jpeg",
                                    use_container_width=True
                                )
                        else:
                            st.error("Composite file not found")
                    else:
                        progress_bar.progress(100)
                        st.error(f"❌ Compositor failed:\n{result.stderr}")
                        st.code(result.stdout, language="text")

                except subprocess.TimeoutExpired:
                    st.error("❌ Compositor timeout (>3 minutes)")
                except Exception as e:
                    st.error(f"❌ Error running compositor: {str(e)}")

# ============================================================
# Footer
# ============================================================
st.divider()
st.markdown("""
---
### About This Tool
This system discovers bare storefronts in London and generates realistic planter visualizations.

**Pipeline:**
1. 🔍 Discover venues using Google Places + Street View + GPT-4 Vision
2. 📦 Extract planters using YOLO + SAM2 segmentation
3. 🎨 Composite onto storefronts with alpha blending

**Design:** [design.md](design.md) | **Code:** [GitHub](https://github.com)
""")
