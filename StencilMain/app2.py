import cv2
import numpy as np
import sys
import argparse

WINDOW_NAME = 'Stencil Preview'


def remove_small_and_thin_features(layer, min_area=1000, min_width=8):
    contours, _ = cv2.findContours(layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(layer)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if area > min_area and w > min_width and h > min_width:
            cv2.drawContours(cleaned, [cnt], -1, 255, thickness=cv2.FILLED)
    return cleaned

def add_and_color_bridges(stencil, bridge_thickness=6):
    num_labels, labels = cv2.connectedComponents(stencil)
    if num_labels <= 2:
        color_preview = cv2.cvtColor(stencil, cv2.COLOR_GRAY2BGR)
        return stencil, color_preview
    centroids = []
    for i in range(1, num_labels):
        mask = (labels == i).astype(np.uint8)
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
    stencil_with_bridges = stencil.copy()
    color_preview = cv2.cvtColor(stencil, cv2.COLOR_GRAY2BGR)
    if len(centroids) > 1:
        for i in range(len(centroids) - 1):
            cv2.line(stencil_with_bridges, centroids[i], centroids[i+1], 255, thickness=bridge_thickness)
            cv2.line(color_preview, centroids[i], centroids[i+1], (0, 0, 255), thickness=bridge_thickness)
    return stencil_with_bridges, color_preview

def gui_mode(image, output_path):
    def nothing(x):
        pass
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Threshold', WINDOW_NAME, 128, 255, nothing)
    cv2.createTrackbar('Min Area', WINDOW_NAME, 1000, 10000, nothing)
    cv2.createTrackbar('Min Width', WINDOW_NAME, 8, 50, nothing)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    while True:
        threshold = cv2.getTrackbarPos('Threshold', WINDOW_NAME)
        min_area = cv2.getTrackbarPos('Min Area', WINDOW_NAME)
        min_width = cv2.getTrackbarPos('Min Width', WINDOW_NAME)
        _, stencil = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        cleaned = remove_small_and_thin_features(stencil, min_area=min_area, min_width=min_width)
        stencil_with_bridges, color_preview = add_and_color_bridges(cleaned)
        cv2.imshow(WINDOW_NAME, color_preview)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite(output_path, stencil_with_bridges)
            print(f"Stencil saved to {output_path} (threshold={threshold}, min_area={min_area}, min_width={min_width})")
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Generate a single stencil image with adjustable threshold and feature size.')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='stencil_output.png', help='Path to save the output stencil image')
    parser.add_argument('--threshold', type=int, default=128, help='Threshold value (0-255)')
    parser.add_argument('--min_area', type=int, default=1000, help='Minimum area for features to keep')
    parser.add_argument('--min_width', type=int, default=8, help='Minimum width/height for features to keep')
    parser.add_argument('--nogui', action='store_true', help='Run in command-line mode (no GUI)')
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not load image from {args.input}")
        sys.exit(1)

    if args.nogui:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, stencil = cv2.threshold(blurred, args.threshold, 255, cv2.THRESH_BINARY)
        cleaned = remove_small_and_thin_features(stencil, min_area=args.min_area, min_width=args.min_width)
        stencil_with_bridges, _ = add_and_color_bridges(cleaned)
        cv2.imwrite(args.output, stencil_with_bridges)
        print(f"Stencil generated and saved to {args.output}")
        print(f"Parameters: threshold={args.threshold}, min_area={args.min_area}, min_width={args.min_width}")
    else:
        gui_mode(image, args.output)

if __name__ == "__main__":
    main() 