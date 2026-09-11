import cv2
import mediapipe as mp
import time
import math
import numpy as np

from filters import FILTERS

MODEL_PATH = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


def pixel_point(landmark, width, height):
    return (
        int(landmark.x * width),
        int(landmark.y * height),
    )


def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denominator = (
        (x1 - x2) * (y3 - y4)
        - (y1 - y2) * (x3 - x4)
    )

    if abs(denominator) < 0.0001:
        return None

    t = (
        (x1 - x3) * (y3 - y4)
        - (y1 - y3) * (x3 - x4)
    ) / denominator

    u = -(
        (x1 - x2) * (y1 - y3)
        - (y1 - y2) * (x1 - x3)
    ) / denominator

    if 0 <= t <= 1 and 0 <= u <= 1:
        collision_x = x1 + t * (x2 - x1)
        collision_y = y1 + t * (y2 - y1)

        return (
            int(collision_x),
            int(collision_y),
        )

    return None


def distance(p1, p2):
    return math.hypot(
        p2[0] - p1[0],
        p2[1] - p1[1]
    )


def apply_filter_to_polygon(frame, points, filter_function, opacity=0.85):
    """
    Apply a filter only inside the polygon defined by points.
    """

    points = np.array(points, dtype=np.int32)

    x, y, w, h = cv2.boundingRect(points)

    x1 = max(x, 0)
    y1 = max(y, 0)

    x2 = min(x + w, frame.shape[1])
    y2 = min(y + h, frame.shape[0])

    if x2 <= x1 or y2 <= y1:
        return frame

    # Extract region
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return frame

    filtered_roi = filter_function(roi)

    local_points = points - np.array([x1, y1])

    # polygon mask
    mask = np.zeros(
        (roi.shape[0], roi.shape[1]),
        dtype=np.uint8
    )

    cv2.fillPoly(
        mask,
        [local_points],
        255
    )

    blended = cv2.addWeighted(
        filtered_roi,
        opacity,
        roi,
        1 - opacity,
        0
    )

    roi[mask > 0] = blended[mask > 0]

    frame[y1:y2, x1:x2] = roi

    return frame


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)

camera = cv2.VideoCapture(0)

current_filter_index = 0

filter_change_cooldown = 1.0
last_filter_change = 0

hands_were_close = False

CLOSE_DISTANCE = 120


with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        ok, frame = camera.read()

        if not ok:
            print("Could not read webcam.")
            break

        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        timestamp_ms = int(
            time.monotonic() * 1000
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        hands_data = []

       # hand detection
        for hand in result.hand_landmarks:

            index = pixel_point(
                hand[8],
                width,
                height
            )

            thumb = pixel_point(
                hand[4],
                width,
                height
            )

            wrist = pixel_point(
                hand[0],
                width,
                height
            )

            cv2.circle(
                frame,
                index,
                10,
                (0, 255, 0),
                -1
            )

            cv2.circle(
                frame,
                thumb,
                10,
                (0, 0, 255),
                -1
            )

            hands_data.append({
                "index": index,
                "thumb": thumb,
                "wrist": wrist
            })

        if len(hands_data) >= 2:

            hand_a = hands_data[0]
            hand_b = hands_data[1]

            ax, ay = hand_a["index"]
            bx, by = hand_b["index"]

            lx, ly = hand_a["thumb"]
            rx, ry = hand_b["thumb"]

            index_a = (ax, ay)
            index_b = (bx, by)

            thumb_a = (lx, ly)
            thumb_b = (rx, ry)

           # rectangle points

            points = np.array(
                [
                    index_a,
                    index_b,
                    thumb_b,
                    thumb_a,
                ],
                dtype=np.int32
            )

            hand_distance = distance(
                index_a,
                index_b
            )

            cv2.line(
                frame,
                index_a,
                index_b,
                (255, 255, 255),
                2
            )

            current_time = time.monotonic()

            hands_are_close = (
                hand_distance < CLOSE_DISTANCE
            )

            if (
                hands_are_close
                and not hands_were_close
                and current_time - last_filter_change
                > filter_change_cooldown
            ):
                current_filter_index += 1

                if current_filter_index >= len(FILTERS):
                    current_filter_index = 0

                last_filter_change = current_time

            # Remember state
            hands_were_close = hands_are_close

            frame = apply_filter_to_polygon(
                frame,
                points,
                FILTERS[current_filter_index],
                opacity=0.85
            )

            cv2.polylines(
                frame,
                [points],
                True,
                (255, 255, 255),
                2
            )

            collision = line_intersection(
                index_a,
                index_b,
                thumb_a,
                thumb_b
            )

            if collision is not None:

                cv2.circle(
                    frame,
                    collision,
                    8,
                    (0, 255, 255),
                    -1
                )

        cv2.putText(
            frame,
            f"Filter: {current_filter_index + 1}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Move hands together to change filter | Q to quit",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Two-Hand Rectangle Filter",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


camera.release()
cv2.destroyAllWindows()
