import cv2
import numpy as np


def filter_1(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    out = np.zeros_like(roi)
    out[gray < 60] = (15, 8, 10)
    out[(gray >= 60) & (gray < 130)] = (118, 30, 214)
    out[(gray >= 130) & (gray < 195)] = (35, 140, 235)
    out[gray >= 195] = (235, 240, 240)
    return out


def filter_2(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cell = 6
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx = (xx % cell) - cell / 2
    cy = (yy % cell) - cell / 2
    dist_center = np.sqrt(cx ** 2 + cy ** 2)
    radius = (1 - gray / 255.0) * (cell / 1.4)
    dot_mask = dist_center < radius
    out = np.full_like(roi, 245)
    out[dot_mask] = (15, 15, 15)
    return out


def filter_3(roi: np.ndarray) -> np.ndarray:
    shift = 6
    b, g, r = cv2.split(roi)
    r_shift = np.roll(r, -shift, axis=1)
    b_shift = np.roll(b, shift, axis=1)
    out = cv2.merge([b_shift, g, r_shift])
    out[::3, :, :] = (out[::3, :, :] * 0.72).astype(np.uint8)
    return out


def filter_5(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def filter_6(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    sepia_kernel = np.array(
        [
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189],
        ]
    )
    sepia = cv2.transform(roi, sepia_kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cy, cx = h / 2, w / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2) or 1.0
    vignette = np.clip(1 - 0.5 * (dist / max_dist), 0, 1)[..., None]

    out = (sepia * vignette).astype(np.uint8)
    noise = np.random.randint(0, 25, out.shape, dtype=np.uint8)
    out = cv2.add(out, noise)
    return out


def filter_white(roi: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(roi, (35, 35), 0)
    white = np.full_like(roi, 255)
    out = cv2.addWeighted(blurred, 0.55, white, 0.45, 0)
    return out


def filter_pink(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cell = 5
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx = (xx % cell) - cell / 2
    cy = (yy % cell) - cell / 2
    dist_center = np.sqrt(cx ** 2 + cy ** 2)
    radius = (1 - gray / 255.0) * (cell / 1.3)
    dot_mask = dist_center < radius

    out = np.full_like(roi, (215, 190, 245))
    out[dot_mask] = (55, 20, 130)
    return out


def filter_grid(roi: np.ndarray) -> np.ndarray:
    out = roi.copy()
    h, w = out.shape[:2]
    step = 22
    color = (235, 235, 235)

    overlay = out.copy()
    for x in range(0, w, step):
        cv2.line(overlay, (x, 0), (x, h), color, 1)
    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w, y), color, 1)

    out = cv2.addWeighted(overlay, 0.75, out, 0.25, 0)
    return out


def filter_fire(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def filter_neon(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    out = np.zeros_like(roi)
    out[edges > 0] = (0, 255, 0)
    return out


def filter_pixel(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (w // 10, h // 10), interpolation=cv2.INTER_LINEAR)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


def filter_invert(roi: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(roi)


def filter_popart(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    out = np.zeros_like(roi)

    half_h, half_w = h // 2, w // 2
    small = cv2.resize(roi, (half_w, half_h))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    out[:half_h, :half_w] = cv2.applyColorMap(gray, cv2.COLORMAP_AUTUMN)
    out[:half_h, half_w:half_w*2] = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
    out[half_h:half_h*2,
        :half_w] = cv2.applyColorMap(gray, cv2.COLORMAP_WINTER)
    out[half_h:half_h*2, half_w:half_w *
        2] = cv2.applyColorMap(gray, cv2.COLORMAP_SPRING)

    return out


FILTERS = [
    filter_grid,
    filter_1,
    filter_2,
    filter_3,
    filter_5,
    filter_6,
    filter_white,
    filter_pink,
    filter_fire,
    filter_neon,
    filter_pixel,
    filter_invert,
    filter_popart,
]
