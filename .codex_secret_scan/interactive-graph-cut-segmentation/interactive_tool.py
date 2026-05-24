import cv2
import numpy as np
import os

from graphcut_core import GraphCutCore, compute_iou


# ---------------------------------------------------------------------
# Global state for GUI
# ---------------------------------------------------------------------

# scribble values (non-zero and distinct!)
BG_VALUE = 65
FG_VALUE = 250

orig_image = None           # original BGR image
display_image = None        # image + scribbles (no segmentation overlay)
scribble_mask = None        # HxW uint8 {0, BG_VALUE, FG_VALUE}
last_segmentation = None    # HxW uint8 {0,255}

drawing = False
prev_x, prev_y = -1, -1

current_label_value = FG_VALUE
brush_radius = 5

undo_stack = []             # list of scribble_mask snapshots
show_overlay = True         # toggle segmentation overlay

current_image_path = None   # used for naming outputs / finding GT


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _refresh_display_image():
    """
    Build display_image from orig_image + scribble_mask.
    """
    global display_image, orig_image, scribble_mask
    if orig_image is None or scribble_mask is None:
        return

    disp = orig_image.copy()

    # BG scribbles in red, FG in green
    bg_mask = scribble_mask == BG_VALUE
    fg_mask = scribble_mask == FG_VALUE

    disp[bg_mask] = (0, 0, 255)   # BGR red
    disp[fg_mask] = (0, 255, 0)   # BGR green

    display_image = disp


def _push_undo_state():
    """
    Store a copy of scribble_mask for undo.
    """
    global undo_stack, scribble_mask
    if scribble_mask is None:
        return
    # limit history length to avoid huge memory usage
    if len(undo_stack) > 50:
        undo_stack.pop(0)
    undo_stack.append(scribble_mask.copy())


def _draw_line(x0, y0, x1, y1):
    """
    Draw scribbles onto scribble_mask (data) and update display_image.
    """
    global scribble_mask, current_label_value

    if scribble_mask is None:
        return

    color_value = int(current_label_value)

    # draw on mask
    cv2.line(
        scribble_mask,
        (x0, y0),
        (x1, y1),
        color_value,
        thickness=brush_radius * 2,
    )

    # re-render display image
    _refresh_display_image()


def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback for drawing scribbles.
    Left button = draw in current FG/BG mode.
    """
    global drawing, prev_x, prev_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        prev_x, prev_y = x, y
        _push_undo_state()          # snapshot before drawing
        _draw_line(x, y, x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        _draw_line(prev_x, prev_y, x, y)
        prev_x, prev_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        _draw_line(prev_x, prev_y, x, y)


def _run_graphcut():
    """
    Run GraphCutCore with current scribbles, update last_segmentation.
    """
    global last_segmentation, orig_image, scribble_mask

    if orig_image is None or scribble_mask is None:
        print("No image or scribbles loaded.")
        return

    # must have at least one FG and one BG scribble
    vals = np.unique(scribble_mask)
    non_zero = vals[vals != 0]
    if non_zero.size < 2:
        print("Need at least one FG and one BG scribble before running Graph Cut.")
        return

    print("Running Graph Cut segmentation ...")

    try:
        segmenter = GraphCutCore(
            color_model_type="gmm",
            lambda_smooth=50.0,
            sigma=10.0,
            connectivity=8,
            use_lab=True,
        )
        last_segmentation = segmenter.segment(orig_image, scribble_mask)
        print("Segmentation updated.")
    except Exception as e:
        print("Error during segmentation:", str(e))


def _maybe_compute_iou_on_save(mask):
    """
    If the image belongs to the provided dataset structure, compute IoU vs GT
    when saving, so you can directly see your interactive performance.
    """
    global current_image_path
    try:
        if current_image_path is None:
            return

        # expect structure: dataset/images/<name>.jpg
        head, img_file = os.path.split(current_image_path)
        head2, parent = os.path.split(head)
        if parent != "images":
            return
        dataset_root = head2
        base = os.path.splitext(img_file)[0]
        gt_path = os.path.join(dataset_root, "images-gt", base + ".png")
        if not os.path.exists(gt_path):
            return

        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            return

        iou = compute_iou(mask, gt_mask)
        print(f"Interactive IoU vs GT for {base}: {iou:.4f}")
    except Exception:
        # don't crash the GUI for any weird path case
        pass


# ---------------------------------------------------------------------
# Dataset picker
# ---------------------------------------------------------------------

def choose_image_from_dataset(dataset_root="dataset"):
    """
    List the images in dataset_root/images and ask the user
    which one to open. Returns full path or None.
    """
    images_dir = os.path.join(dataset_root, "images")
    if not os.path.isdir(images_dir):
        print(f"No dataset images directory found at: {images_dir}")
        return None

    files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not files:
        print(f"No images found in {images_dir}")
        return None

    print("\nAvailable dataset images:")
    for i, name in enumerate(files):
        print(f"  [{i}] {name}")

    print("\nType the index of the image you want to open")
    print("(or just press ENTER for index 0, 'q' to quit)")

    while True:
        s = input("Your choice: ").strip()
        if s == "":
            idx = 0
            break
        if s.lower() == "q":
            return None
        if not s.isdigit():
            print("Please enter a valid index number or 'q'.")
            continue
        idx = int(s)
        if 0 <= idx < len(files):
            break
        print(f"Index out of range [0..{len(files)-1}]. Try again.")

    chosen = files[idx]
    full_path = os.path.join(images_dir, chosen)
    print(f"Selected image: {chosen}")
    return full_path


# ---------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------

def run_interactive_tool(image_path):
    """
    Launch interactive segmentation tool on given image.

    Controls:
      Mouse:
        - Left button: draw scribbles (FG or BG depending on mode)

      Keyboard:
        - f : foreground scribbles (green)
        - b : background scribbles (red)
        - + : increase brush size
        - - : decrease brush size
        - u or SPACE : run Graph Cut with current scribbles
        - o : toggle overlay (scribbles vs scribbles+segmentation)
        - z : undo last scribble operation
        - r : reset scribbles and segmentation
        - s : save current segmentation mask
        - q or ESC : quit
    """
    global orig_image, display_image, scribble_mask, last_segmentation
    global current_label_value, brush_radius, undo_stack, show_overlay, current_image_path

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    current_image_path = image_path
    orig_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if orig_image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    h, w = orig_image.shape[:2]
    scribble_mask = np.zeros((h, w), dtype=np.uint8)
    last_segmentation = None
    undo_stack = []
    show_overlay = True
    current_label_value = FG_VALUE
    brush_radius = 5

    _refresh_display_image()

    window_name = "Interactive Graph Cut - Image"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n=== Interactive Graph Cut Tool ===")
    print(f"Image: {image_path}")
    print("Controls:")
    print("  Mouse:")
    print("    Left button : draw scribbles")
    print("  Keys:")
    print("    f : foreground mode (green)")
    print("    b : background mode (red)")
    print("    + : increase brush size")
    print("    - : decrease brush size")
    print("    u or SPACE : update segmentation (run Graph Cut)")
    print("    o : toggle overlay (scribbles vs scribbles+segmentation)")
    print("    z : undo last scribble")
    print("    r : reset scribbles and segmentation")
    print("    s : save current segmentation mask")
    print("    q or ESC : quit")
    print("=================================\n")

    while True:
        # build visualization: scribbles (+ optional seg overlay)
        if display_image is None:
            vis = orig_image.copy()
        else:
            vis = display_image.copy()

        if last_segmentation is not None and show_overlay:
            overlay = vis.copy()
            fg = last_segmentation > 0
            overlay[fg] = (
                0.4 * overlay[fg].astype(np.float32)
                + 0.6 * np.array([0, 0, 255], dtype=np.float32)
            ).astype(np.uint8)
            vis = overlay

        cv2.imshow(window_name, vis)
        key = cv2.waitKey(20) & 0xFF

        # quit
        if key in (27, ord("q")):
            break

        # foreground mode
        elif key == ord("f"):
            current_label_value = FG_VALUE
            print("Mode: FOREGROUND (green)")

        # background mode
        elif key == ord("b"):
            current_label_value = BG_VALUE
            print("Mode: BACKGROUND (red)")

        # increase brush
        elif key == ord("+"):
            brush_radius = min(brush_radius + 1, 50)
            print(f"Brush radius: {brush_radius}")

        # decrease brush
        elif key == ord("-"):
            brush_radius = max(1, brush_radius - 1)
            print(f"Brush radius: {brush_radius}")

        # toggle overlay
        elif key == ord("o"):
            show_overlay = not show_overlay
            print(f"Overlay: {'ON' if show_overlay else 'OFF'}")

        # undo
        elif key == ord("z"):
            if undo_stack:
                last = undo_stack.pop()
                scribble_mask[:] = last
                _refresh_display_image()
                print("Undo: restored previous scribble state.")
            else:
                print("Undo: no history.")

        # reset
        elif key == ord("r"):
            scribble_mask[:] = 0
            last_segmentation = None
            undo_stack = []
            _refresh_display_image()
            print("Reset scribbles and segmentation.")

        # run graph cut
        elif key in (ord("u"), 32):  # 'u' or SPACE
            _run_graphcut()

        # save segmentation
        elif key == ord("s"):
            if last_segmentation is None:
                print("No segmentation to save yet.")
            else:
                base = os.path.splitext(os.path.basename(image_path))[0]
                out_name = base + "_interactive_mask.png"
                cv2.imwrite(out_name, last_segmentation)
                print(f"Saved segmentation mask to: {out_name}")
                _maybe_compute_iou_on_save(last_segmentation)

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # user gave a path explicitly
        img_path = sys.argv[1]
    else:
        # let user choose from dataset/images
        print("No image path provided on command line.")
        img_path = choose_image_from_dataset(dataset_root="dataset")
        if img_path is None:
            print("No image selected. Exiting.")
            sys.exit(0)

    run_interactive_tool(img_path)
