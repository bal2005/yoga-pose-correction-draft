import cv2
from time import time
import pickle as pk
import mediapipe as mp
import pandas as pd
import pyttsx4
import multiprocessing as mtp
from calc_angles import rangles
from landmarks import extract_landmarks
from recommendations import check_pose_angle

# Initialize Video Capture


def init_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file!")
        exit()
    return cap

# Get Pose Name


def get_pose_name(index):
    names = {
        0: "Adho Mukha Svanasana",
        1: "Phalakasana",
        2: "Utkata Konasana",
        3: "Vrikshasana",
    }
    return str(names.get(index, "Unknown Pose"))

# Initialize Landmark and Column Names


def init_dicts():
    landmarks_points = {
        "nose": 0,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28,
        "left_heel": 29, "right_heel": 30,
        "left_foot_index": 31, "right_foot_index": 32,
    }
    col_names = []
    for name in landmarks_points.keys():
        col_names += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_v"]
    landmarks_points_array = {key: [] for key in landmarks_points.keys()}
    return col_names, landmarks_points_array

# Text-to-Speech Process


def tts(tts_q):
    engine = pyttsx4.init()
    while True:
        objects = tts_q.get()
        if objects is None:
            break
        message = objects[0]
        engine.say(message)
        engine.runAndWait()
    tts_q.task_done()

# Display Text on Video


def cv2_put_text(image, message):
    cv2.putText(
        image, message, (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5, cv2.LINE_AA
    )

# Release Resources


def destroy(cap, tts_proc, tts_q):
    cv2.destroyAllWindows()
    cap.release()
    tts_q.put(None)
    tts_q.close()
    tts_q.join_thread()
    tts_proc.join()


# Main Function
if __name__ == "__main__":
    # Replace with the path to your video file
    video_path = "sample2.mp4"
    cap = init_video(video_path)

    # Load Pre-trained Model and CSV Data
    model = pk.load(open("./models/poses.model", "rb"))
    cols, landmarks_points_array = init_dicts()
    angles_df = pd.read_csv("./csv_files/poses_angles.csv")
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialize Text-to-Speech Queue
    tts_q = mtp.JoinableQueue()
    tts_proc = mtp.Process(target=tts, args=(tts_q,))
    tts_proc.start()

    tts_last_exec = time() + 5  # To manage audio feedback timing

    while True:
        result, image = cap.read()
        if not result:
            print("End of video.")
            destroy(cap, tts_proc, tts_q)
            break

        # Preprocess Image
        flipped = cv2.flip(image, 1)
        resized_image = cv2.resize(
            flipped, (640, 360), interpolation=cv2.INTER_AREA)

        # Exit on 'q'
        key = cv2.waitKey(1)
        if key == ord("q"):
            destroy(cap, tts_proc, tts_q)
            break

        # Landmark Extraction and Pose Prediction
        err, df, landmarks = extract_landmarks(resized_image, mp_pose, cols)
        if not err:
            prediction = model.predict(df)
            probabilities = model.predict_proba(df)

            # Draw Landmarks
            mp_drawing.draw_landmarks(
                flipped, landmarks, mp_pose.POSE_CONNECTIONS
            )

            # Check Confidence Threshold
            if probabilities[0, prediction[0]] > 0.85:  # Adjusted confidence threshold
                pose_name = get_pose_name(prediction[0])
                cv2_put_text(flipped, pose_name)
                tts_q.put(["Perfect"])

                # Calculate Angles and Provide Suggestions
                angles = rangles(df, landmarks_points_array)
                suggestions = check_pose_angle(
                    prediction[0], angles, angles_df)

                # Audio Feedback
                if time() > tts_last_exec:
                    if not suggestions:  # If no suggestions, the pose is perfect
                        tts_q.put(["Perfect"])
                    else:  # Provide the first suggestion
                        tts_q.put([suggestions[0]])
                    tts_last_exec = time() + 5
            else:
                cv2_put_text(flipped, "No Pose Detected")
        else:
            print(extract_landmarks(resized_image, mp_pose, cols))
            cv2_put_text(flipped, "Error in Processing Frame")

        # Display Video
        cv2.imshow("Yoga Pose Detection", flipped)
