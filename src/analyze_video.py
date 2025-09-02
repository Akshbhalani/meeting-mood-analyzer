import cv2
from fer import FER
import csv
import datetime
import os

# Initialize detector
detector = FER(mtcnn=True)

# Ensure the output folder exists
output_folder = "../data/video"
os.makedirs(output_folder, exist_ok=True)

# Create CSV file with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = os.path.join(output_folder, f"emotion_log_{timestamp}.csv")

# Open webcam
video_capture = cv2.VideoCapture(0)

with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Frame", "Dominant Emotion", "Confidence"])  # header row

    frame_count = 0
    print("Press 'q' to quit video analysis...")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect emotions
        result = detector.detect_emotions(frame)

        if result:
            emotions = result[0]["emotions"]
            dominant_emotion = max(emotions, key=emotions.get)
            confidence = emotions[dominant_emotion]

            # Write to CSV
            writer.writerow([frame_count, dominant_emotion, round(confidence, 2)])

            # Show on frame
            cv2.putText(frame, f"{dominant_emotion} ({confidence:.2f})",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display video
        cv2.imshow("Video Emotion Analyzer", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

# Release resources
video_capture.release()
cv2.destroyAllWindows()

print(f"Emotion log saved to: {csv_filename}")
