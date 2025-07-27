import cv2
import numpy as np
import argparse
import os

WINDOW_NAME = 'Color Segmentation Preview'

def remove_small_and_thin_features(layer, min_area=1000, min_width=8):
    contours, _ = cv2.findContours(layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(layer)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if area > min_area and w > min_width and h > min_width:
            cv2.drawContours(cleaned, [cnt], -1, 255, thickness=cv2.FILLED)
    return cleaned

def segment_by_color(image, n_colors=4):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    segmented = res.reshape((image.shape))
    masks = []
    for i in range(n_colors):
        mask = (label.reshape(image.shape[:2]) == i).astype(np.uint8) * 255
        masks.append(mask)
    return masks, segmented, center

def gui_preview(image, masks, center, output_dir):
    def nothing(x):
        pass
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Min Area', WINDOW_NAME, 1000, 10000, nothing)
    cv2.createTrackbar('Min Width', WINDOW_NAME, 8, 50, nothing)
    n_colors = len(masks)
    while True:
        min_area = cv2.getTrackbarPos('Min Area', WINDOW_NAME)
        min_width = cv2.getTrackbarPos('Min Width', WINDOW_NAME)
        # Overlay each cleaned mask in its color
        overlay = np.zeros_like(image)
        cleaned_masks = []
        for i, mask in enumerate(masks):
            cleaned = remove_small_and_thin_features(mask, min_area=min_area, min_width=min_width)
            cleaned_masks.append(cleaned)
            color = [int(c) for c in center[i]]
            overlay[cleaned == 255] = color
        cv2.imshow(WINDOW_NAME, overlay)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            os.makedirs(output_dir, exist_ok=True)
            for i, cleaned in enumerate(cleaned_masks):
                out_path = os.path.join(output_dir, f'layer_color_{i+1:02d}.png')
                cv2.imwrite(out_path, cleaned)
                print(f"Saved: {out_path}")
            print(f"All layers saved to {output_dir}/ (min_area={min_area}, min_width={min_width})")
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Segment image by color and generate stencil layers for each color.')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='color_layers', help='Directory to save output layers')
    parser.add_argument('--n_colors', type=int, default=4, help='Number of color segments/layers')
    parser.add_argument('--nogui', action='store_true', help='Skip preview window')
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not load image from {args.input}")
        return
    masks, segmented, center = segment_by_color(image, n_colors=args.n_colors)
    if not args.nogui:
        gui_preview(image, masks, center, args.output_dir)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        for i, mask in enumerate(masks):
            cleaned = remove_small_and_thin_features(mask)
            out_path = os.path.join(args.output_dir, f'layer_color_{i+1:02d}.png')
            cv2.imwrite(out_path, cleaned)
            print(f"Saved: {out_path}")
        print(f"Generated {len(masks)} color-based stencil layers in {args.output_dir}/")

if __name__ == "__main__":
    main() 