# Stabilized Face Recognition System

This Python project implements a real-time face recognition system with enhanced stability and accuracy. It leverages **OpenCV** for video capture and basic face detection, **DeepFace** for robust face encoding, and a custom **FaceTracker** class for smoothing and stabilizing face detections across frames. The system can learn new faces interactively and stores known face data for future recognition.

---

## Features

* **Real-time Face Detection:** Utilizes Haar Cascades with optimized parameters for robust initial face detection.
* **Face Tracking and Stabilization:** A custom `FaceTracker` class smooths bounding box movements and increases confidence in detected faces over multiple frames, reducing jitters and false positives.
* **Deep Learning-based Face Recognition:** Employs **DeepFace** (specifically the FaceNet model) to generate high-dimensional face embeddings for accurate recognition.
* **Interactive Unknown Face Learning:** When an unknown, stable face is detected, a pop-up dialog (Tkinter) prompts the user to enter the person's name, allowing for on-the-fly enrollment.
* **Persistent Data Storage:** Known face encodings and names are saved using `pickle` and `json` files, respectively, ensuring that learned faces are remembered across sessions.
* **Recognition Caching:** Caches recognition results for stable faces to avoid redundant processing, improving performance.
* **Visual Feedback:** Displays bounding boxes, recognized names, confidence levels, and stability indicators on the live video feed.

---

## Prerequisites

Before running the system, ensure you have the following installed:

* Python 3.7+
* OpenCV
* DeepFace
* TensorFlow (a backend for DeepFace)
* Scikit-learn (for `cosine_similarity`)
* Tkinter (usually comes pre-installed with Python)

You can install the necessary Python packages using pip:

```bash
pip install opencv-python deepface tensorflow scikit-learn
```

**Note:** DeepFace can be resource-intensive. A GPU can significantly speed up the embedding generation process if TensorFlow is configured to use it.

---

## How to Run

1.  **Save the Code:** Save the provided Python code as a `.py` file (e.g., `face_recognition_system.py`).
2.  **Run from Terminal:** Open your terminal or command prompt, navigate to the directory where you saved the file, and run:

    ```bash
    python face_recognition_system.py
    ```
3.  **Interact:**
    * The system will start capturing video from your default webcam.
    * Faces will be detected and tracked.
    * If a stable, unknown face is detected, a dialog box will appear asking for the person's name.
        * Enter a name and press "OK" to add the face to the database.
        * Press "Cancel" or close the dialog to skip adding the face.
    * Recognized faces will be labeled with their names and confidence scores.
    * Press 'q' to quit the application.

---

## Project Structure and Key Components

* **`FaceTracker` Class:**
    * Manages the state of detected faces across frames.
    * Uses a **smoothing factor** to create stable bounding boxes.
    * `min_confidence_frames`: Defines how many consecutive frames a face must be detected to be considered "stable" for recognition processing.
    * `_find_best_match` and `_calculate_overlap`: Help in associating new detections with existing tracked faces using Intersection Over Union (IOU).
    * `processed_unknown_faces`: A set to ensure that the "ask for name" dialog for a specific unknown face only appears once.

* **`FaceRecognitionSystem` Class:**
    * **Initialization:** Loads pre-existing face data (`face_data.pkl`, `face_names.json`). Initializes OpenCV's Haar Cascade for face detection and sets up the video capture.
    * **`load_face_data()` / `save_face_data()`:** Handles the serialization and deserialization of face encodings and names.
    * **`get_face_encoding(face_img)`:** The core function that uses `DeepFace.represent()` to get a 128-dimensional embedding for a given face image. Includes validation for face image quality (size, brightness, pixel variation).
    * **`recognize_face(face_encoding)`:** Compares a new face encoding against known encodings using **cosine similarity** to find the best match.
    * **`detect_faces_stable(frame)`:** Improves upon basic Haar Cascade detection by applying image preprocessing (histogram equalization, Gaussian blur) and stricter filtering parameters (`minNeighbors`, `minSize`, `maxSize`, aspect ratio checks) to reduce false positives.
    * **`ask_for_name()`:** A Tkinter-based function to prompt the user for input when an unknown face is encountered. Runs in the main thread to avoid UI issues.
    * **`add_new_face(face_encoding, name)`:** Adds a new face's encoding and name to the database and saves the updated data.
    * **`recognition_cache`:** A dictionary to store recent recognition results for `face_id`s, preventing redundant DeepFace calls.
    * **`should_process_unknown_face()`:** Implements logic to decide when an unknown face is sufficiently stable and "unknown" enough to warrant asking the user for a name.
    * **`run_recognition()`:** The main loop:
        * Captures frames.
        * Calls `detect_faces_stable` to get initial detections.
        * Updates the `face_tracker`.
        * Periodically (every `recognition_interval` frames) processes stable faces for recognition using cached results or by computing new embeddings.
        * If a stable, unknown face is detected and hasn't been processed before, it triggers the `ask_for_name` dialog.
        * Draws bounding boxes and labels on the frame.

---

## Customization and Tuning

* **`FaceTracker` Parameters:**
    * `smoothing_factor` (default: 0.8): Higher values lead to smoother, less responsive bounding boxes. Lower values make them more reactive to movement.
    * `min_confidence_frames` (default: 3): Number of frames a face must be consistently detected before it's considered stable. Increase this for higher stability, decrease for faster recognition of new faces.

* **`FaceRecognitionSystem` Parameters:**
    * `confidence_threshold` (default: 0.7): The cosine similarity score above which a face is considered recognized. Adjust this based on your desired strictness. Higher values mean fewer false positives but potentially more "Unknown" classifications.
    * `cache_timeout` (default: 3.0 seconds): How long a recognition result is cached for a specific tracked face. Increase this for more performance but potentially slower updates if someone changes identity mid-stream (unlikely for a single person).
    * `recognition_interval` (default: 8 frames): How often DeepFace recognition is performed on stable faces. Increasing this reduces CPU/GPU load but might make recognition appear slightly less responsive.
    * **`detectMultiScale` parameters:** Experiment with `scaleFactor`, `minNeighbors`, `minSize`, `maxSize` in `detect_faces_stable` for your specific lighting conditions and camera setup.
    * **`should_process_unknown_face` logic:** The current logic requires `confidence >= 15` (very stable) and `confidence < 0.45` (very low similarity to known faces). Adjust these thresholds if you want to be more or less aggressive in prompting for unknown faces.

---

## Troubleshooting

* **"Error loading face data"**: This is usually harmless on the first run as `face_data.pkl` and `face_names.json` won't exist yet. They will be created when you add the first face.
* **"Make sure you have installed..."**: If you see this, double-check your `pip install` commands.
* **Slow performance / High CPU/GPU usage:** DeepFace can be computationally intensive.
    * Ensure your TensorFlow installation is optimized for your hardware (e.g., using GPU if available).
    * Increase `recognition_interval`.
    * Reduce `CAP_PROP_FPS` in `self.video_capture.set()`.
* **Poor detection or recognition:**
    * Ensure good lighting conditions.
    * Try adjusting the Haar Cascade parameters in `detect_faces_stable()`.
    * Make sure the faces are reasonably sized and clear when adding them to the database.
    * Adjust `confidence_threshold`.
* **Tkinter dialog not appearing or freezing:** Tkinter dialogs need to run in the main thread. The current implementation tries to handle this, but if you're embedding this into a larger application, be mindful of threading issues.

---

Feel free to experiment with the parameters to optimize performance and accuracy for your specific environment!
