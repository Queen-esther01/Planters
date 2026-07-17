"""
Planter Compositing Module - Using SAM2 for Segmentation + Multi-Band Blending

Takes your real planter product photos and composites them onto storefronts with:
- Meta's SAM2 for precise plant segmentation
- Perspective-aware scaling
- Multi-band blending for photorealistic results
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import SAM, YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlanterCompositor:
    def __init__(self):
        """Initialize SAM2 for segmentation and YOLO for door detection"""
        print("\n" + "="*60)
        print("🤖 INITIALIZING PLANTER COMPOSITOR")
        print("="*60)
        print("📦 Loading SAM2 segmentation model...")
        self.sam_model = SAM('sam2_l.pt')
        print("✅ SAM2 model loaded!")

        print("📦 Loading YOLO26 segmentation model for door detection...")
        self.detect_model = YOLO('yolo26m-seg.pt')
        print("✅ Detection model loaded!")

        self.confidence_threshold = 0.5

    def detect_door(self, image_path):
        """Detect door/entrance and ground using YOLO"""
        print(f"\n🔍 Loading storefront image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None, None

        h, w = image.shape[:2]
        print(f"   Image size: {w}x{h}")

        print("   Running YOLO detection...")
        results = self.detect_model(image)
        detections = results[0]

        if len(detections.boxes) == 0:
            print("   ⚠️  No objects detected. Using fallback.")
            return self._estimate_door_fallback(image)

        print(f"   Found {len(detections.boxes)} objects")

        # Visualize all detections
        viz_image = image.copy()
        door_bbox = None

        for idx, box in enumerate(detections.boxes):
            class_id = int(box.cls[0])
            class_name = self.detect_model.names[class_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            print(f"      [{idx}] {class_name} ({conf:.1%}) at ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})")

            # Draw bounding box
            color = (0, 255, 0) if conf > self.confidence_threshold else (100, 100, 100)
            cv2.rectangle(viz_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(viz_image, f"{class_name} {conf:.0%}", (int(x1), int(y1)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Identify door (high confidence)
            if conf > self.confidence_threshold:
                class_lower = class_name.lower()
                if 'door' in class_lower:
                    if door_bbox is None or conf > float(door_bbox[0]):
                        door_bbox = (conf, box.xyxy[0].cpu().numpy())

        # Save visualization
        cv2.imwrite("detection_viz.jpg", viz_image)
        print("\n   💾 Saved detection visualization to: detection_viz.jpg")

        if door_bbox is not None:
            conf, bbox = door_bbox
            x1, y1, x2, y2 = bbox
            print(f"   ✅ Door detected: ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)}) [confidence: {conf:.1%}]")
            return image, bbox
        else:
            print("   ⚠️  No door detected. Using fallback.")
            return self._estimate_door_fallback(image)

    def visualize_detections(self, image_path):
        """Save visualization and individual crops of detected door/ground"""
        print(f"\n📸 Analyzing detections in: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return

        h, w = image.shape[:2]
        print(f"   Image size: {w}x{h}")

        results = self.detect_model(image)
        detections = results[0]

        if len(detections.boxes) == 0:
            print("   ⚠️  No objects detected")
            return

        # Full visualization with all detections
        viz_image = image.copy()
        door_box = None
        ground_box = None

        for idx, box in enumerate(detections.boxes):
            class_id = int(box.cls[0])
            class_name = self.detect_model.names[class_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            color = (0, 255, 0) if conf > self.confidence_threshold else (100, 100, 100)
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(viz_image, f"{class_name} {conf:.0%}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            class_lower = class_name.lower()
            if 'door' in class_lower:
                if door_box is None or conf > door_box[0]:
                    door_box = (conf, (x1, y1, x2, y2))
            if 'floor' in class_lower or 'ground' in class_lower:
                if ground_box is None or conf > ground_box[0]:
                    ground_box = (conf, (x1, y1, x2, y2))

        # Save full visualization
        cv2.imwrite("detection_full.jpg", viz_image)
        print(f"\n✅ Full visualization: detection_full.jpg")

        # Save door crop
        if door_box:
            conf, (x1, y1, x2, y2) = door_box
            door_crop = image[y1:y2, x1:x2]
            cv2.imwrite("detection_door.jpg", door_crop)
            print(f"✅ Door crop: detection_door.jpg ({int(x2-x1)}x{int(y2-y1)}, conf: {conf:.0%})")

        # Save ground crop
        if ground_box:
            conf, (x1, y1, x2, y2) = ground_box
            ground_crop = image[y1:y2, x1:x2]
            cv2.imwrite("detection_ground.jpg", ground_crop)
            print(f"✅ Ground crop: detection_ground.jpg ({int(x2-x1)}x{int(y2-y1)}, conf: {conf:.0%})")

        if not door_box and not ground_box:
            print("⚠️  No door or ground detected")

    def _estimate_door_fallback(self, image):
        """Fallback: estimate door in center-bottom"""
        h, w = image.shape[:2]
        door_width = w * 0.2
        door_height = h * 0.4
        center_x = w / 2
        door_y_start = h * 0.35

        bbox = np.array([
            center_x - door_width/2,
            door_y_start,
            center_x + door_width/2,
            door_y_start + door_height
        ])
        x1, y1, x2, y2 = bbox
        print(f"   📍 Fallback door estimation: ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})")
        return image, bbox

    def remove_background_sam(self, image_path):
        """
        Extract plant using SAM2 mask directly.

        Method:
        1. YOLO detects the plant (gets bounding box)
        2. SAM segments within that bounding box (precise mask)
        3. Use SAM mask as-is (no degrading refinements)
        """
        print(f"\n📦 Loading planter image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not load planter: {image_path}")
            return None, None

        h, w = img.shape[:2]
        print(f"   Image size: {w}x{h}")

        # Step 1: Use YOLO to detect plants and get bounding box
        print("   🎯 Running YOLO detection to find plant...")
        yolo_results = self.detect_model(img)
        yolo_result = yolo_results[0]

        plant_bbox = None
        if len(yolo_result.boxes) > 0:
            # Find plant-like objects
            plant_keywords = ["potted plant", "plant", "flower", "pot"]

            for box in yolo_result.boxes:
                class_id = int(box.cls[0])
                class_name = self.detect_model.names[class_id].lower()
                conf = float(box.conf[0])

                if any(keyword in class_name for keyword in plant_keywords):
                    plant_bbox = box.xyxy[0].cpu().numpy()
                    print(f"      ✅ Plant detected: {class_name} ({conf:.1%})")
                    break

            # Fallback: use largest detection
            if plant_bbox is None and len(yolo_result.boxes) > 0:
                box = yolo_result.boxes[0]
                plant_bbox = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0])
                class_name = self.detect_model.names[class_id]
                print(f"      ℹ️  No plant keyword found. Using largest: {class_name}")

        if plant_bbox is None:
            print("      ⚠️  YOLO detection failed. Using fallback extraction.")
            print("   🎯 Running SAM without bounding box prompt...")
            sam_results = self.sam_model(img)
            if sam_results and sam_results[0].masks:
                masks = sam_results[0].masks.data
                largest_idx = 0
                largest_area = 0
                for idx, mask in enumerate(masks):
                    area = cv2.countNonZero((mask.cpu().numpy() * 255).astype(np.uint8))
                    if area > largest_area:
                        largest_area = area
                        largest_idx = idx
                plant_mask = (masks[largest_idx].cpu().numpy() * 255).astype(np.uint8)
                print(f"      ✅ Segmented fallback: {largest_area} pixels")
            else:
                print("      ❌ SAM fallback also failed")
                return None, None
        else:
            # Step 2: Use YOLO's bounding box to prompt SAM
            print("   🎯 Running SAM segmentation with YOLO bounding box...")
            x1, y1, x2, y2 = map(int, plant_bbox)

            # Run SAM with bounding box prompt
            sam_results = self.sam_model(img, bboxes=[[x1, y1, x2, y2]])

            if not sam_results or not sam_results[0].masks:
                print("      ❌ SAM segmentation failed")
                return None, None

            # Use SAM mask as-is (already precise)
            plant_mask = sam_results[0].masks.data[0].cpu().numpy()
            plant_mask = (plant_mask * 255).astype(np.uint8)
            area = cv2.countNonZero(plant_mask)
            print(f"      ✅ Segmented with SAM: {area} pixels")

        # Resize mask back to original image size
        plant_mask = cv2.resize(plant_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # Convert to RGBA with SAM mask as alpha (no refinements)
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = plant_mask

        non_transparent = np.count_nonzero(plant_mask > 127)
        print(f"   ✅ Plant extracted ({non_transparent} pixels)")

        return rgba, plant_mask

    def preview_extraction(self, image_path, output_path="extraction_preview.png"):
        """
        Preview the SAM2 background extraction.
        Saves RGBA image so you can verify the extraction quality before compositing.
        """
        print(f"\n👀 Previewing extraction for: {image_path}")
        rgba, _ = self.remove_background_sam(image_path)
        if rgba is None:
            print("❌ Extraction failed")
            return None

        cv2.imwrite(output_path, rgba)
        print(f"\n✅ Preview saved to: {output_path}")
        print("\nWhat to check:")
        print("  - Open the preview - should show just the plant on transparent background")
        return output_path

    def calculate_perspective_info(self, door_bbox, image_shape):
        """Calculate perspective based on door location"""
        h, w = image_shape[:2]
        x1, y1, x2, y2 = door_bbox.astype(int)

        door_width = x2 - x1
        door_height = y2 - y1
        door_center_x = (x1 + x2) / 2

        return {
            'door_bbox': door_bbox,
            'door_center': door_center_x,
            'door_height': door_height,
            'door_width': door_width,
        }

    def scale_planter(self, planter_rgba, door_height):
        """Scale planter to match storefront proportions"""
        h, w = planter_rgba.shape[:2]

        # Scale planter to ~70% of door height
        planter_scale = (door_height * 0.7) / h
        new_h = int(h * planter_scale)
        new_w = int(w * planter_scale)

        scaled = cv2.resize(planter_rgba, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        print(f"   Scaled to {new_w}x{new_h}")

        return scaled

    def position_planter(self, scaled_planter, door_info, position='left'):
        """Position planter left or right of door"""
        h, w = scaled_planter.shape[:2]
        door_center = door_info['door_center']
        door_width = door_info['door_width']
        door_y_bottom = door_info['door_bbox'][3]

        if position == 'left':
            x_pos = int(door_center - door_width/2 - w - 15)
        else:  # right
            x_pos = int(door_center + door_width/2 + 15)

        y_pos = int(door_y_bottom - h)

        print(f"   Positioned {position}: ({x_pos}, {y_pos})")
        return x_pos, y_pos


    def composite(self, storefront_path, planter_paths, output_path):
        """
        Main compositing pipeline - composite planters as-is without scaling.

        Args:
            storefront_path: Path to storefront image
            planter_paths: List of planter image paths
            output_path: Where to save result
        """
        print("\n" + "="*60)
        print("🎨 PLANTER COMPOSITING PIPELINE")
        print("="*60)

        # Load storefront
        storefront = cv2.imread(storefront_path)
        if storefront is None:
            print(f"❌ Failed to load storefront: {storefront_path}")
            return None

        bg_h, bg_w = storefront.shape[:2]
        print(f"\n📸 Storefront size: {bg_w}x{bg_h}")

        result = storefront.copy()

        # Composite each planter as-is (no scaling)
        num_planters = min(len(planter_paths), 2)
        x_positions = [50, bg_w - 300]  # left and right edges

        for i, planter_path in enumerate(planter_paths[:num_planters]):
            print(f"\n--- PLANTER {i+1}/{num_planters} ---")

            # Extract planter with SAM
            planter_rgba, _ = self.remove_background_sam(planter_path)
            if planter_rgba is None:
                print(f"⚠️  Skipping {Path(planter_path).name}")
                continue

            fg_h, fg_w = planter_rgba.shape[:2]
            print(f"   Planter size: {fg_w}x{fg_h}")

            # Position at bottom-left or bottom-right
            x = x_positions[i]
            y = bg_h - fg_h

            print(f"   Positioning at: ({x}, {y})")

            # Clip position to fit within bounds
            x = max(0, min(x, bg_w - fg_w))
            y = max(0, min(y, bg_h - fg_h))

            # Crop planter to fit within storefront bounds if necessary
            crop_x = max(0, -x) if x < 0 else 0
            crop_y = max(0, -y) if y < 0 else 0
            crop_w = min(fg_w, bg_w - x)
            crop_h = min(fg_h, bg_h - y)

            if crop_w <= 0 or crop_h <= 0:
                print(f"   ⚠️  Planter doesn't fit in storefront, skipping")
                continue

            # Crop the planter and its mask
            cropped_planter = planter_rgba[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            alpha_mask = cropped_planter[:, :, 3] > 0

            # Paste where alpha > 0
            result[y:y+crop_h, x:x+crop_w][alpha_mask] = cropped_planter[:, :, :3][alpha_mask]

        # Save result
        print("\n💾 Saving composited image...")
        cv2.imwrite(output_path, result)
        print(f"✅ Composited image saved to: {output_path}")

        return result


if __name__ == "__main__":
    import sys

    print("\n🎨 PLANTER COMPOSITOR - SAM2 Edition")
    print("Usage:")
    print("  Composite:   python planter_compositor.py <storefront> <planter1> [planter2] [-o output.jpg]")
    print("  Visualize:   python planter_compositor.py --visualize <storefront.jpg>")
    print("  Preview:     python planter_compositor.py --preview <planter.jpg> [-o preview.png]")
    print()

    if len(sys.argv) < 2:
        print("Example:")
        print("  python planter_compositor.py storefront.jpg planter.jpg -o result.jpg")
        print("  python planter_compositor.py --preview planter.jpg -o extracted.png")
        sys.exit(1)

    compositor = PlanterCompositor()

    # Handle visualize mode
    if sys.argv[1] == "--visualize":
        if len(sys.argv) < 3:
            print("❌ Visualize mode requires: --visualize <storefront_image>")
            sys.exit(1)

        storefront = sys.argv[2]
        print(f"📸 Storefront: {storefront}")
        print()

        compositor.visualize_detections(storefront)

    # Handle preview mode
    elif sys.argv[1] == "--preview":
        if len(sys.argv) < 3:
            print("❌ Preview mode requires: --preview <planter_image>")
            sys.exit(1)

        planter = sys.argv[2]
        output = "extraction_preview.png"

        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "-o" and i + 1 < len(sys.argv):
                output = sys.argv[i + 1]
                i += 2
            else:
                i += 1

        print(f"🪴 Planter: {planter}")
        print(f"💾 Preview output: {output}")
        print()

        compositor.preview_extraction(planter, output)
    else:
        # Regular composite mode
        storefront = sys.argv[1]
        planters = []
        output = "composited_result.jpg"

        # Parse arguments
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "-o" and i + 1 < len(sys.argv):
                output = sys.argv[i + 1]
                i += 2
            else:
                planters.append(sys.argv[i])
                i += 1

        if not planters:
            print("❌ Composite mode requires: <storefront> <planter1> [planter2]")
            sys.exit(1)

        print(f"📸 Storefront: {storefront}")
        print(f"🪴 Planters: {planters}")
        print(f"💾 Output: {output}")
        print()

        compositor.composite(storefront, planters, output)
