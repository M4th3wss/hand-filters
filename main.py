import cv2
import mediapipe as mp
import time
import math
import numpy as np

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
        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    )

    if abs(denominator) < 0.0001:
        return None

    t = (
        (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    ) / denominator

    u = -(
        (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
    ) / denominator

    # t and u are inside line segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        collision_x = x1 + t * (x2 - x1)
        collision_y = y1 + t * (y2 - y1)
        return (int(collision_x), int(collision_y))

    return None


def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)

camera = cv2.VideoCapture(0)

# Line's saved endpoints
line_x1, line_y1 = 0, 0
line_x2, line_y2 = 0, 0

line_x3, line_y3 = 0, 0
line_x4, line_y4 = 0, 0


with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = camera.read()

        if not ok:
            print("Could not read webcam.")
            break

        # Mirror the camera like a selfie view
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # OpenCV uses BGR; MediaPipe needs RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        # VIDEO mode requires timestamps that always increase
        timestamp_ms = int(time.monotonic() * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        hands_data = []

        # Detect, draw, and collect data for every visible hand
        for hand in result.hand_landmarks:
            index = pixel_point(hand[8], width, height)
            thumb = pixel_point(hand[4], width, height)

            cv2.circle(frame, index, 10, (0, 255, 0), -1)
            cv2.circle(frame, thumb, 10, (0, 0, 255), -1)

            pinch_distance = distance(index, thumb)
            is_pinching = pinch_distance < 45

            hands_data.append({
                "index": index,
                "thumb": thumb,
            })
        if len(hands_data) >= 2:
            hand_a = hands_data[0]
            hand_b = hands_data[1]

            ax, ay = hand_a["index"]  # left index x, y
            bx, by = hand_b["index"]  # right index x, y

            lx, ly = hand_a["thumb"]  # left thumb x, y
            rx, ry = hand_b["thumb"]  # right thumb x, y

            line_x1, line_y1 = ax, ay
            line_x2, line_y2 = bx, by
            line_x3, line_y3 = lx, ly
            line_x4, line_y4 = rx, ry

            index_line_start = (ax, ay)
            index_line_end = (bx, by)

            thumb_line_start = (lx, ly)
            thumb_line_end = (rx, ry)
            collision = line_intersection(
                index_line_start,
                index_line_end,
                thumb_line_start,
                thumb_line_end,
            )
            points = np.array([
                [line_x1, line_y1],
                [line_x2, line_y2],
                [line_x4, line_y4],
                [line_x3, line_y3],

            ], dtype=np.int32)

            overlay = frame.copy()

            cv2.fillPoly(overlay, [points], (255, 80, 0))
            frame = cv2.addWeighted(overlay, 0.40, frame, 0.5, 0)
            cv2.polylines(frame, [points], True, (255, 255, 255), 2)

            if collision is not None:
                collision_x, collision_y = collision
                # Triangle between left hand's index, thumb, and crossing point
                left_area = np.array([
                    (ax, ay),             # left index
                    (lx, ly),             # left thumb
                    collision,
                ], dtype=np.int32)

                # Triangle between right hand's index, thumb, and crossing point
                right_area = np.array([
                    (bx, by),             # right index
                    (rx, ry),             # right thumb
                    collision,
                ], dtype=np.int32)

                # Draw transparent fills on an overlay
                overlay = frame.copy()

                # BGR colors
                cv2.fillPoly(overlay, [left_area], (255, 80, 0))     # blue
                cv2.fillPoly(overlay, [right_area],
                             (0, 80, 255))    # red/orange

                # Blend the colors with the webcam image
                frame = cv2.addWeighted(overlay, 0.40, frame, 0.60, 0)

                # Draw the area outlines
                cv2.polylines(frame, [left_area], True, (255, 255, 255), 2)
                cv2.polylines(frame, [right_area], True, (255, 255, 255), 2)
        cv2.putText(
            frame,
            "Pinch both hands to resize | Q to quit",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Two-Hand Rectangle Resize", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
