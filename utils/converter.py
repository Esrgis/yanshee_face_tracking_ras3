"""Label converter utilities.

Converts common ground-truth formats (YOLO txt, COCO json) into
per-image bbox files with lines in the format:

	x y w h category_id score

Where x,y are top-left pixel coordinates, w/h are width/height in pixels.

Usage (CLI):
  python -m utils.converter --input-type yolo --labels labels_dir --images images_dir --out out_dir
  python -m utils.converter --input-type coco --coco coco.json --out out_dir

This file intentionally keeps dependencies minimal (uses OpenCV only to read image sizes).
"""

from __future__ import annotations

import os
import json
import glob
import argparse
from collections import defaultdict
from typing import List, Tuple

try:
	import cv2
except Exception:
	cv2 = None


def ensure_dir(path: str):
	os.makedirs(path, exist_ok=True)


def write_bbox_file(path: str, bboxes: List[Tuple[float, float, float, float, int, float]], out_format: str = 'minimal'):
	"""Write bboxes to file.

	Formats supported:
	  - 'minimal': each line `x y w h` (integers) — compatible with Vision bbox usage
	  - 'full':    each line `x y w h category_id score`
	"""
	with open(path, 'w') as f:
		for x, y, w, h, cid, score in bboxes:
			if out_format == 'full':
				f.write(f"{int(round(x))} {int(round(y))} {int(round(w))} {int(round(h))} {int(cid)} {float(score):.3f}\n")
			else:
				f.write(f"{int(round(x))} {int(round(y))} {int(round(w))} {int(round(h))}\n")


def yolo_line_to_bbox(tokens: List[str], img_w: int, img_h: int):
	"""Convert one YOLO-format line to bbox (x,y,w,h,category,score).

	YOLO expected formats:
	  class cx cy w h          (normalized)
	  class cx cy w h conf     (normalized, optional confidence)
	All values are floats (cx,cy,w,h relative to image size)
	Returns None for malformed lines.
	"""
	if len(tokens) < 5:
		return None
	try:
		cid = int(tokens[0])
		cx = float(tokens[1])
		cy = float(tokens[2])
		rw = float(tokens[3])
		rh = float(tokens[4])
		score = float(tokens[5]) if len(tokens) >= 6 else 1.0
	except ValueError:
		return None

	# Convert normalized center to pixel top-left + width/height
	w = rw * img_w
	h = rh * img_h
	cx_px = cx * img_w
	cy_px = cy * img_h
	x = cx_px - w / 2.0
	y = cy_px - h / 2.0
	# clamp
	x = max(0.0, min(x, img_w - 1))
	y = max(0.0, min(y, img_h - 1))
	w = max(1.0, min(w, img_w - x))
	h = max(1.0, min(h, img_h - y))
	return (x, y, w, h, cid, score)


def validate_out_format(out_format: str) -> bool:
	"""Validate output format choice."""
	if out_format not in ('minimal', 'full'):
		raise ValueError(f"Unsupported out_format: {out_format}")
	return True


def validate_yolo_label_file(path: str, max_lines: int = 10) -> bool:
	"""Quick validation of a YOLO label file.

	Checks a few lines to ensure each line has at least 5 numeric tokens
	and that normalized coords are within [0,1] (warn if not).
	Returns True if basic checks pass, False otherwise.
	"""
	try:
		with open(path, 'r') as f:
			for i, ln in enumerate(f):
				if i >= max_lines:
					break
				ln = ln.strip()
				if not ln:
					continue
				toks = ln.split()
				if len(toks) < 5:
					print(f"[ERROR] YOLO label malformed (fewer than 5 tokens): {path}")
					return False
				# try parse
				try:
					int(toks[0])
					nums = list(map(float, toks[1:5]))
				except Exception:
					print(f"[ERROR] YOLO label contains non-numeric values: {path}")
					return False
				# check normalized range (common YOLO uses normalized coords)
				for v in nums:
					if v < 0.0 or v > 1.0:
						print(f"[WARN] YOLO coords appear outside [0,1] in {path}; file may use absolute pixels")
						# still accept but warn
						break
		return True
	except FileNotFoundError:
		print(f"[ERROR] YOLO label file not found: {path}")
		return False


def validate_yolo_dir(labels_dir: str, images_dir: str) -> bool:
	"""Validate labels and images directories exist and contain matching basenames."""
	if not os.path.isdir(labels_dir):
		print(f"[ERROR] Labels directory not found: {labels_dir}")
		return False
	if not os.path.isdir(images_dir):
		print(f"[ERROR] Images directory not found: {images_dir}")
		return False
	label_paths = glob.glob(os.path.join(labels_dir, '*.txt'))
	if not label_paths:
		print(f"[ERROR] No .txt label files found in {labels_dir}")
		return False
	# check first few label files
	for lp in label_paths[:5]:
		if not validate_yolo_label_file(lp):
			return False
	return True


def validate_coco_json_file(coco_json_path: str) -> bool:
	"""Validate COCO json file structure minimally."""
	if not os.path.isfile(coco_json_path):
		print(f"[ERROR] COCO json not found: {coco_json_path}")
		return False
	try:
		with open(coco_json_path, 'r') as f:
			data = json.load(f)
	except Exception as e:
		print(f"[ERROR] Failed to parse COCO json: {e}")
		return False
	if 'images' not in data or 'annotations' not in data:
		print(f"[ERROR] COCO json missing required keys 'images' or 'annotations'")
		return False
	# basic check of annotations
	for ann in data.get('annotations', [])[:10]:
		if 'image_id' not in ann or 'bbox' not in ann:
			print(f"[ERROR] COCO annotation missing image_id or bbox: {ann}")
			return False
	return True


def convert_yolo_dir(labels_dir: str, images_dir: str, out_dir: str, out_format: str = 'minimal'):
	"""Convert a directory of YOLO .txt labels to per-image bbox txt files.

	labels_dir: directory containing .txt files (same basename as images)
	images_dir: directory containing image files to read sizes
	out_dir: directory to write bbox files
	"""
	# Validate inputs and requested output format
	validate_out_format(out_format)
	if not validate_yolo_dir(labels_dir, images_dir):
		return
	ensure_dir(out_dir)
	# Find all label files
	label_paths = glob.glob(os.path.join(labels_dir, '*.txt'))
	if not label_paths:
		print(f"[WARN] No label .txt found in {labels_dir}")
		return

	# Build quick image index by basename
	image_index = defaultdict(list)
	for img_ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
		for p in glob.glob(os.path.join(images_dir, img_ext)):
			base = os.path.splitext(os.path.basename(p))[0]
			image_index[base].append(p)

	for lp in label_paths:
		base = os.path.splitext(os.path.basename(lp))[0]
		# try to find a matching image
		img_paths = image_index.get(base, [])
		if not img_paths:
			print(f"[WARN] No image matched for label {lp}, skipping")
			continue
		img_path = img_paths[0]

		if cv2:
			img = cv2.imread(img_path)
			if img is None:
				print(f"[WARN] Failed to read image {img_path}, skipping")
				continue
			h, w = img.shape[:2]
		else:
			print("[ERROR] OpenCV not available. Provide image sizes via other means.")
			return

		bboxes = []
		with open(lp, 'r') as f:
			for ln in f:
				ln = ln.strip()
				if not ln:
					continue
				toks = ln.split()
				converted = yolo_line_to_bbox(toks, w, h)
				if converted:
					bboxes.append(converted)

		out_path = os.path.join(out_dir, base + '.txt')
		write_bbox_file(out_path, bboxes, out_format=out_format)
		print(f"[OK] Wrote {len(bboxes)} bboxes → {out_path}")


def convert_coco_json(coco_json_path: str, images_dir: str | None, out_dir: str, out_format: str = 'minimal'):
	"""Convert COCO annotation json to per-image bbox txt files.

	coco_json_path: path to COCO format JSON
	images_dir: optional images dir used to verify filenames; images names are taken from JSON
	out_dir: directory to write bbox files
	"""
	validate_out_format(out_format)
	if not validate_coco_json_file(coco_json_path):
		return
	ensure_dir(out_dir)
	with open(coco_json_path, 'r') as f:
		data = json.load(f)

	images = {img['id']: img for img in data.get('images', [])}
	anns_per_image = defaultdict(list)
	for ann in data.get('annotations', []):
		img_id = ann['image_id']
		bbox = ann.get('bbox')  # COCO bbox: [x,y,w,h] in pixels
		if bbox is None:
			continue
		cid = ann.get('category_id', 0)
		score = ann.get('score', 1.0)
		x, y, w, h = bbox
		anns_per_image[img_id].append((x, y, w, h, cid, score))

	for img_id, bboxes in anns_per_image.items():
		img_info = images.get(img_id)
		if img_info is None:
			continue
		fname = img_info.get('file_name')
		base = os.path.splitext(os.path.basename(fname))[0]
		out_path = os.path.join(out_dir, base + '.txt')
		write_bbox_file(out_path, bboxes, out_format=out_format)
		print(f"[OK] Wrote {len(bboxes)} bboxes → {out_path}")


def main():
	p = argparse.ArgumentParser(description='Convert labels to bbox txt files (x y w h category score)')
	p.add_argument('--input-type', choices=['yolo', 'coco'], required=True)
	p.add_argument('--labels', help='Directory with YOLO .txt labels (for yolo)')
	p.add_argument('--images', help='Directory with images (needed to get sizes for yolo)')
	p.add_argument('--coco', help='COCO json file (for coco)')
	p.add_argument('--out', required=True, help='Output directory for bbox txts')
	p.add_argument('--out-format', choices=['minimal', 'full'], default='minimal', help='Output bbox format')
	args = p.parse_args()

	if args.input_type == 'yolo':
		if not args.labels or not args.images:
			p.error('--labels and --images are required for YOLO conversion')
		convert_yolo_dir(args.labels, args.images, args.out, out_format=args.out_format)
	else:
		if not args.coco:
			p.error('--coco is required for COCO conversion')
		convert_coco_json(args.coco, args.images, args.out, out_format=args.out_format)


if __name__ == '__main__':
	main()

