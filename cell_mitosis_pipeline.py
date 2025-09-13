"""
cell_mitosis_pipeline.py
----------------------------------------------
Quick-and-dirty nuclei-extraction + DHE-Mit-Classifier inference.

Requirements
------------
• Python ≥ 3.7
• OpenCV-python, scikit-image, numpy, torch 1.7, torchvision 0.8
• (optional) openslide-python for .svs or .tiff WSI files

Run
---
$ python cell_mitosis_pipeline.py slide.svs \
      --net ASTMNet --weights trained_models/ASTMNet.ckpt \
      --output_csv results/predictions.csv
"""

import argparse, os, sys, csv, pathlib, warnings
import cv2, numpy as np
from skimage import color, morphology, measure, segmentation, filters, exposure
import torch
import torchvision.transforms as T

# ---- import the CNN definition --------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT / "py_files"))
from ASTMNet import ASTMNet         # <- change if you want another net

# ---------------------------------------------------------------------------

def load_slide(path, level=0):
    """Return an RGB numpy image.
       Uses OpenSlide if a multi-resolution file, else cv2.imread."""
    try:
        import openslide
        slide = openslide.OpenSlide(path)
        img = slide.read_region((0, 0), level, slide.level_dimensions[level])
        img = cv2.cvtColor(np.array(img)[:, :, :3], cv2.COLOR_RGB2BGR)
    except (ImportError, openslide.OpenSlideError, FileNotFoundError):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image {path}")
    return img

# ----------------------- nuclei / cell segmentation ------------------------
def segment_nuclei(rgb_img):
    """
    Very simple segmentation:
    1. Color-deconvolve H&E → take hematoxylin (nuclei) channel
    2. Otsu threshold + morpho clean-up
    3. Distance-transform + watershed to split clumps
    Returns: list of bounding boxes [(x, y, w, h), …]
    """
    # 1. colour deconvolution
    ihc_hed = color.rgb2hed(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    hema = exposure.rescale_intensity(-ihc_hed[..., 0], out_range=(0, 1))
    
    # 2. threshold & post-processing
    th = filters.threshold_otsu(hema)
    bw = hema > th
    bw = morphology.remove_small_objects(bw, 50)
    bw = morphology.remove_small_holes(bw, 50)
    bw = morphology.binary_opening(bw, morphology.disk(2))
    
    # 3. watershed to separate touching nuclei
    dist = morphology.distance_transform_edt(bw)
    coords = morphology.peak_local_max(dist, footprint=np.ones((3, 3)), labels=bw)
    markers = np.zeros_like(dist, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    labels = segmentation.watershed(-dist, markers, mask=bw)
    
    # 4. bounding boxes
    boxes = []
    for region in measure.regionprops(labels):
        y, x, h, w = region.bbox[0], region.bbox[1], \
                     region.bbox[2] - region.bbox[0], region.bbox[3] - region.bbox[1]
        if 15 < h < 200 and 15 < w < 200:        # size filter
            boxes.append((x, y, w, h))
    return boxes

# ----------------------- CNN utilities -------------------------------------
def load_model(net_name="ASTMNet", weight_path="trained_models/ASTMNet.ckpt",
               device="cuda" if torch.cuda.is_available() else "cpu"):
    net = ASTMNet() if net_name == "ASTMNet" else None  # extend for other nets
    if net is None:
        raise ValueError(f"Unknown net {net_name}")
    net.load_state_dict(torch.load(weight_path, map_location=device))
    net.eval().to(device)
    return net, device


def patch_to_tensor(patch_bgr):
    """120×120 to tensor with original training normalisation."""
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    patch_rgb = cv2.resize(patch_rgb, (120, 120), interpolation=cv2.INTER_CUBIC)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats (used in paper)
                    std=[0.229, 0.224, 0.225]),
    ])
    return transform(patch_rgb)                   # C×H×W tensor


def infer_patches(net, device, patches):
    """Return list of (prob_mitosis, predicted_label)."""
    with torch.no_grad():
        batch = torch.stack([patch_to_tensor(p) for p in patches]).to(device)
        logits = net(batch)
        probs = torch.softmax(logits, dim=1)[:, 1]     # class 1 = mitosis
        labels = (probs > 0.5).long().cpu().numpy()
        return probs.cpu().numpy(), labels

# --------------------------- main routine ----------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slide", help="./slides/.slide (1).png")
    parser.add_argument("--net", default="ASTMNet")
    parser.add_argument("--weights", default="trained_models/ASTMNet.ckpt")
    parser.add_argument("--output_csv", default="predictions.csv")
    args = parser.parse_args()

    # 1. read slide & segment nuclei
    img = load_slide(args.slide)
    boxes = segment_nuclei(img)
    print(f"[INFO] found {len(boxes)} candidate nuclei")

    if not boxes:
        print("[WARN] No nuclei detected – exiting.")
        return

    # 2. crop patches
    patches, coords = [], []
    for (x, y, w, h) in boxes:
        cx, cy = x + w // 2, y + h // 2
        half = 60                                           # 120×120 patch
        x0, y0 = max(cx - half, 0), max(cy - half, 0)
        x1, y1 = min(cx + half, img.shape[1] - 1), min(cy + half, img.shape[0] - 1)
        patch = img[y0:y1, x0:x1].copy()
        patches.append(patch)
        coords.append((cx, cy))

    # 3. run CNN
    model, device = load_model(args.net, args.weights)
    probs, labels = infer_patches(model, device, patches)

    # 4. save CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_id", "x", "y", "prob_mitosis", "pred_label"])
        for i, ((x, y), p, l) in enumerate(zip(coords, probs, labels)):
            writer.writerow([i, x, y, f"{p:.4f}", int(l)])
    print(f"[INFO] results written to {args.output_csv}")

    # 5. optional overlay for quick visual check
    overlay = img.copy()
    for (x, y), l in zip(coords, labels):
        colour = (0, 0, 255) if l else (0, 255, 0)  # red = mitosis
        cv2.circle(overlay, (x, y), 20, colour, 2)
    viz_path = pathlib.Path(args.output_csv).with_suffix(".overlay.jpg")
    cv2.imwrite(str(viz_path), overlay)
    print(f"[INFO] overlay saved to {viz_path}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")   # suppress sk-image & sklearn spam
    main()
# This script is a quick-and-dirty pipeline for nuclei segmentation and mitosis classification.