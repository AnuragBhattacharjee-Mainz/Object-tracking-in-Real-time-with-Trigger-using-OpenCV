import cv2
import numpy as np
import math
import time

def create_skin_mask(frame):
    
    blurred = cv2.GaussianBlur(frame, (9, 9), 0)

    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)

    mask = cv2.inRange(ycrcb, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)

    return mask


def get_largest_contour(mask, min_area=1000):
  
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, 0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < min_area:
        return None, 0

    return largest, area


def get_contour_centroid(contour):
    
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def distance_point_to_rect(point, rect):

    px, py = point
    x1, y1, x2, y2 = rect

    cx = max(x1, min(px, x2))
    cy = max(y1, min(py, y2))

    dx = px - cx
    dy = py - cy
    distance = math.sqrt(dx * dx + dy * dy)
    return distance

def distance_contour_to_rect(contour, rect):

    min_dist = float('inf')
    for pt in contour:
        x, y = pt[0]
        d = distance_point_to_rect((x, y), rect)
        if d < min_dist:
            min_dist = d
    return min_dist

def smooth_point(prev_point, new_point, alpha=0.7):
   
    if prev_point is None:
        return new_point
    if new_point is None:
        return prev_point

    px, py = prev_point
    nx, ny = new_point

    sx = int(alpha * px + (1 - alpha) * nx)
    sy = int(alpha * py + (1 - alpha) * ny)
    return (sx, sy)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

virtual_rect = None 

SAFE_DISTANCE = 70     
WARNING_DISTANCE = 30 
DETECTION_MAX_DISTANCE = 200
MAX_HAND_AREA_RATIO = 0.18

prev_centroid = None
prev_time = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    if virtual_rect is None:
        rect_w, rect_h = 150, 150
        x1 = w // 2 - rect_w // 2
        y1 = h // 2 - rect_h // 2
        x2 = x1 + rect_w
        y2 = y1 + rect_h
        virtual_rect = (x1, y1, x2, y2)

    mask = create_skin_mask(frame)

    hand_contour, area = get_largest_contour(mask)

    hand_centroid = None
    distance = None

    state = "NO_HAND"

    if hand_contour is not None:
        frame_area = w * h
        area_ratio = area / float(frame_area)
        if area_ratio > MAX_HAND_AREA_RATIO:
         hand_contour = None
         hand_centroid = None
        else:
         hand_centroid = get_contour_centroid(hand_contour)
        
         if hand_centroid is not None:
          cx, cy = hand_centroid
        
          if cy < int(0.50 * h):
            hand_contour = None
            hand_centroid = None

    if hand_centroid is not None:
        hand_centroid = smooth_point(prev_centroid, hand_centroid, alpha=0.7)
    prev_centroid = hand_centroid

    if hand_contour is not None:
      if hand_centroid is not None:
        approx_dist = distance_point_to_rect(hand_centroid, virtual_rect)
      else:
        approx_dist = float('inf')
      if approx_dist > DETECTION_MAX_DISTANCE:
        # Too far from the virtual object – treat as NO_HAND
        hand_contour = None
        hand_centroid = None
        distance = None
        state = "NO_HAND"
      else:
        # Close enough: use the more precise contour-based distance
        distance = distance_contour_to_rect(hand_contour, virtual_rect)
        if distance > SAFE_DISTANCE:
            state = "SAFE"
        elif distance > WARNING_DISTANCE:
            state = "WARNING"
        else:
            state = "DANGER"

    else:
     distance = None
     state = "NO_HAND"


    x1, y1, x2, y2 = virtual_rect

    if state == "DANGER":
        rect_color = (0, 0, 255)  
    elif state == "WARNING":
        rect_color = (0, 255, 255)
    else:
        rect_color = (0, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 3)

    if hand_contour is not None:
        cv2.drawContours(frame, [hand_contour], -1, (255, 0, 0), 2)

    if hand_centroid is not None:
        cx, cy = hand_centroid
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(frame, "Hand", (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if distance is not None:
        cv2.putText(frame, f"Distance: {int(distance)} px", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Distance: N/A", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    if state == "SAFE" or state =="NO_HAND":
        state_text = "SAFE"
        state_color = (0, 255, 0)
    elif state == "WARNING":
        state_text = "WARNING"
        state_color = (0, 255, 255)
    elif state == "DANGER":
        state_text = "DANGER DANGER"
        state_color = (0, 0, 255)


    cv2.putText(frame, f"STATE: {state_text}", (10, 80),
              cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 3)

    curr_time = time.time()
    dt = curr_time - prev_time
    prev_time = curr_time
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Hand Safety System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()