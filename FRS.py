import cv2
import numpy as np
import pickle
import os
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog, messagebox
import threading
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import json
from deepface import DeepFace
import warnings
warnings.filterwarnings('ignore')

class FaceTracker:
    """Class to track and stabilize face detection"""
    def __init__(self, smoothing_factor=0.7, min_confidence_frames=5):
        self.smoothing_factor = smoothing_factor
        self.min_confidence_frames = min_confidence_frames
        self.tracked_faces = {}
        self.next_face_id = 0
        self.processed_unknown_faces = set()  # Track faces we've already processed
        
    def update_faces(self, detected_faces, frame_shape):
        """Update tracked faces with new detections"""
        current_faces = {}
        
        for (x, y, w, h) in detected_faces:
            # Find best matching tracked face
            best_match_id = self._find_best_match(x, y, w, h)
            
            if best_match_id is not None:
                # Update existing face
                face_data = self.tracked_faces[best_match_id]
                
                # Smooth the coordinates
                face_data['x'] = int(self.smoothing_factor * face_data['x'] + (1 - self.smoothing_factor) * x)
                face_data['y'] = int(self.smoothing_factor * face_data['y'] + (1 - self.smoothing_factor) * y)
                face_data['w'] = int(self.smoothing_factor * face_data['w'] + (1 - self.smoothing_factor) * w)
                face_data['h'] = int(self.smoothing_factor * face_data['h'] + (1 - self.smoothing_factor) * h)
                
                face_data['confidence'] += 1
                face_data['last_seen'] = datetime.now()
                
                current_faces[best_match_id] = face_data
            else:
                # Create new tracked face
                face_data = {
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'confidence': 1,
                    'last_seen': datetime.now(),
                    'recognition_name': None,
                    'recognition_confidence': 0.0,
                    'stable': False,
                    'processed_for_unknown': False  # Track if this face was processed for unknown
                }
                current_faces[self.next_face_id] = face_data
                self.next_face_id += 1
        
        # Remove old faces and their processed status
        current_time = datetime.now()
        faces_to_keep = {}
        for face_id, face_data in current_faces.items():
            if (current_time - face_data['last_seen']).total_seconds() < 1.0:
                faces_to_keep[face_id] = face_data
            else:
                # Remove from processed set when face disappears
                self.processed_unknown_faces.discard(face_id)
        
        self.tracked_faces = faces_to_keep
        return self.get_stable_faces()
    
    def _find_best_match(self, x, y, w, h):
        """Find the best matching tracked face"""
        best_match_id = None
        best_overlap = 0
        
        for face_id, face_data in self.tracked_faces.items():
            overlap = self._calculate_overlap(x, y, w, h, 
                                            face_data['x'], face_data['y'], 
                                            face_data['w'], face_data['h'])
            if overlap > 0.3 and overlap > best_overlap:  # 30% overlap threshold
                best_overlap = overlap
                best_match_id = face_id
        
        return best_match_id
    
    def _calculate_overlap(self, x1, y1, w1, h1, x2, y2, w2, h2):
        """Calculate overlap ratio between two rectangles"""
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left < right and top < bottom:
            intersection = (right - left) * (bottom - top)
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            return intersection / union if union > 0 else 0
        return 0
    
    def get_stable_faces(self):
        """Return only stable faces"""
        stable_faces = []
        for face_id, face_data in self.tracked_faces.items():
            if face_data['confidence'] >= self.min_confidence_frames:
                face_data['stable'] = True
                stable_faces.append((face_id, face_data))
        return stable_faces
    
    def update_recognition(self, face_id, name, confidence):
        """Update recognition results for a face"""
        if face_id in self.tracked_faces:
            self.tracked_faces[face_id]['recognition_name'] = name
            self.tracked_faces[face_id]['recognition_confidence'] = confidence
    
    def mark_processed_for_unknown(self, face_id):
        """Mark that we already processed this face for unknown"""
        self.processed_unknown_faces.add(face_id)
        if face_id in self.tracked_faces:
            self.tracked_faces[face_id]['processed_for_unknown'] = True
    
    def is_processed_for_unknown(self, face_id):
        """Check if this face was already processed for unknown"""
        return face_id in self.processed_unknown_faces

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.confidence_threshold = 0.7
        self.data_file = "face_data.pkl"
        self.names_file = "face_names.json"
        
        # Initialize face tracker
        self.face_tracker = FaceTracker(smoothing_factor=0.8, min_confidence_frames=3)
        
        # Load existing data
        self.load_face_data()
        
        # Initialize face detector with better parameters
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize video capture
        self.video_capture = cv2.VideoCapture(0)
        
        # Set camera properties for better stability
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video_capture.set(cv2.CAP_PROP_FPS, 30)
        
        # Recognition cache to avoid repeated processing
        self.recognition_cache = {}
        self.cache_timeout = 3.0  # seconds - increased for more stability
        
        # Track unknown faces to prevent repeated dialogs
        self.unknown_faces_processed = set()
        
        print("Stabilized Face Recognition System initialized")
        
    def load_face_data(self):
        """Load previously saved face encodings and names"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    
            if os.path.exists(self.names_file):
                with open(self.names_file, 'r') as f:
                    self.known_face_names = json.load(f)
                    
            print(f"Loaded {len(self.known_face_encodings)} faces from database")
                    
        except Exception as e:
            print(f"Error loading face data: {e}")
            self.known_face_encodings = []
            self.known_face_names = []
    
    def save_face_data(self):
        """Save face encodings and names to files"""
        try:
            with open(self.data_file, 'wb') as f:
                pickle.dump({'encodings': self.known_face_encodings}, f)
            
            with open(self.names_file, 'w') as f:
                json.dump(self.known_face_names, f)
                
            print("Face data saved successfully")
                
        except Exception as e:
            print(f"Error saving face data: {e}")
    
    def get_face_encoding(self, face_img):
        """Extract face encoding using DeepFace with error handling and validation"""
        try:
            # Ensure minimum face size
            if face_img.shape[0] < 80 or face_img.shape[1] < 80:
                return None
            
            # Check if image has reasonable content (not just noise or uniform color)
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
            
            # Check for reasonable variation in pixel values (faces should have some variation)
            pixel_std = np.std(gray_face)
            if pixel_std < 10:  # Too uniform, likely not a real face
                return None
            
            # Check for reasonable brightness (not too dark or too bright)
            mean_brightness = np.mean(gray_face)
            if mean_brightness < 30 or mean_brightness > 220:
                return None
                
            # Resize face for consistency
            face_resized = cv2.resize(face_img, (160, 160))
            
            # Convert to RGB
            if len(face_resized.shape) == 3:
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_resized
            
            # Get face embedding using DeepFace with stricter detection
            embedding = DeepFace.represent(face_rgb, 
                                         model_name='Facenet',
                                         enforce_detection=True)
            
            if isinstance(embedding, list) and len(embedding) > 0:
                return np.array(embedding[0]['embedding'])
            else:
                return np.array(embedding['embedding'])
                
        except Exception as e:
            # If DeepFace fails to detect a face, it's likely not a real face
            return None
    
    def recognize_face(self, face_encoding):
        """Recognize face using cosine similarity"""
        if len(self.known_face_encodings) == 0:
            return "Unknown", 0.0
        
        try:
            # Calculate cosine similarities
            similarities = []
            for known_encoding in self.known_face_encodings:
                similarity = cosine_similarity([face_encoding], [known_encoding])[0][0]
                similarities.append(similarity)
            
            # Find best match
            max_similarity = max(similarities)
            best_match_idx = similarities.index(max_similarity)
            
            if max_similarity > self.confidence_threshold:
                return self.known_face_names[best_match_idx], max_similarity
            else:
                return "Unknown", max_similarity
                
        except Exception as e:
            return "Unknown", 0.0
    
    def detect_faces_stable(self, frame):
        """Detect faces with improved parameters for stability and accuracy"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection
        gray = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Detect faces with more conservative parameters to reduce false positives
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,        # Larger scale factor for more stability
            minNeighbors=15,        # Much higher threshold to reduce false positives
            minSize=(80, 80),       # Larger minimum size to avoid small objects
            maxSize=(400, 400),     # Add maximum size to avoid very large false detections
            flags=cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_DO_CANNY_PRUNING
        )
        
        # Additional filtering to remove obvious non-faces
        filtered_faces = []
        for (x, y, w, h) in faces:
            # Check aspect ratio (faces should be roughly square to rectangular)
            aspect_ratio = w / h
            if 0.6 <= aspect_ratio <= 1.4:  # Reasonable face aspect ratio
                # Check if face is not too close to edges (often false positives)
                if (x > 20 and y > 20 and 
                    x + w < frame.shape[1] - 20 and 
                    y + h < frame.shape[0] - 20):
                    filtered_faces.append((x, y, w, h))
        
        return filtered_faces
    
    def ask_for_name(self):
        """Show dialog to ask for person's name - run in main thread"""
        try:
            root = tk.Tk()
            root.withdraw()  # Hide the main window immediately
            root.attributes('-topmost', True)  # Make window appear on top
            root.lift()  # Bring to front
            root.focus_force()  # Force focus
            
            name = simpledialog.askstring("Unknown Face Detected", 
                                         "Who is this person?\n(Press Cancel to skip)",
                                         parent=root)
            root.quit()  # Properly quit the tkinter mainloop
            root.destroy()  # Then destroy the window
            return name
        except Exception as e:
            print(f"Error showing dialog: {e}")
            return None
    
    def add_new_face(self, face_encoding, name):
        """Add new face to the database"""
        if name and name.strip():
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name.strip())
            self.save_face_data()
            print(f"Added new face: {name}")
            return True
        return False
    
    def get_cached_recognition(self, face_id):
        """Get cached recognition result"""
        current_time = datetime.now().timestamp()
        if (face_id in self.recognition_cache and 
            current_time - self.recognition_cache[face_id]['timestamp'] < self.cache_timeout):
            cache_data = self.recognition_cache[face_id]
            return cache_data['name'], cache_data['confidence']
        return None, None
    
    def cache_recognition(self, face_id, name, confidence):
        """Cache recognition result"""
        self.recognition_cache[face_id] = {
            'name': name,
            'confidence': confidence,
            'timestamp': datetime.now().timestamp()
        }
    
    def should_process_unknown_face(self, face_id, face_data, confidence):
        """Determine if we should process this unknown face"""
        # Don't process if already processed
        if self.face_tracker.is_processed_for_unknown(face_id):
            return False
        
        # Only process very stable faces with low confidence
        if (face_data['confidence'] >= 15 and  # Face must be very stable
            confidence < 0.45 and              # Very low similarity to known faces
            face_data['stable']):              # Must be marked as stable
            return True
        
        return False
    
    def run_recognition(self):
        """Main recognition loop with stabilization"""
        print("Starting stabilized face recognition system...")
        print("Press 'q' to quit")
        
        frame_count = 0
        recognition_interval = 8  # Process recognition every 8 frames for stability
        
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            frame_count += 1
            
            # Detect faces every frame for smooth tracking
            detected_faces = self.detect_faces_stable(frame)
            
            # Update face tracker
            stable_faces = self.face_tracker.update_faces(detected_faces, frame.shape)
            
            # Process recognition less frequently
            if frame_count % recognition_interval == 0:
                for face_id, face_data in stable_faces:
                    if not face_data['stable']:
                        continue
                    
                    # Check cache first
                    cached_name, cached_conf = self.get_cached_recognition(face_id)
                    if cached_name is not None:
                        self.face_tracker.update_recognition(face_id, cached_name, cached_conf)
                        continue
                    
                    # Extract face region
                    x, y, w, h = face_data['x'], face_data['y'], face_data['w'], face_data['h']
                    
                    # Add padding and ensure bounds
                    padding = 20
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    
                    face_img = frame[y1:y2, x1:x2]
                    
                    if face_img.size > 0:
                        # Get face encoding
                        face_encoding = self.get_face_encoding(face_img)
                        
                        if face_encoding is not None:
                            # Recognize face
                            name, confidence = self.recognize_face(face_encoding)
                            
                            # Cache the result
                            self.cache_recognition(face_id, name, confidence)
                            
                            # Update face tracker
                            self.face_tracker.update_recognition(face_id, name, confidence)
                            
                            # Handle unknown faces - only process once per face
                            if (name == "Unknown" and 
                                self.should_process_unknown_face(face_id, face_data, confidence)):
                                
                                print(f"Processing unknown face {face_id} (confidence: {confidence:.3f})")
                                
                                # Mark as processed BEFORE asking for name
                                self.face_tracker.mark_processed_for_unknown(face_id)
                                
                                # Ask for name
                                user_name = self.ask_for_name()
                                
                                if user_name and user_name.strip():
                                    # Add new face to database
                                    if self.add_new_face(face_encoding, user_name):
                                        # Clear recognition cache to update immediately
                                        self.recognition_cache = {}
                                        print(f"Successfully added {user_name} to database")
                                        # Update the face tracker with new name immediately
                                        self.face_tracker.update_recognition(face_id, user_name, 1.0)
                                else:
                                    print("Skipped adding unknown face")
            
            # Draw all stable faces
            for face_id, face_data in stable_faces:
                if not face_data['stable']:
                    continue
                
                x, y, w, h = face_data['x'], face_data['y'], face_data['w'], face_data['h']
                name = face_data.get('recognition_name', 'Processing...')
                confidence = face_data.get('recognition_confidence', 0.0)
                
                # Choose color based on recognition status
                if name == 'Processing...':
                    color = (255, 255, 0)  # Yellow for processing
                elif name == "Unknown":
                    color = (0, 0, 255)    # Red for unknown
                else:
                    color = (0, 255, 0)    # Green for recognized
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw label background
                label = f"{name}"
                if name != 'Processing...' and confidence > 0:
                    label += f" ({confidence:.2f})"
                
                # Add stability indicator
                stability_text = f"S:{face_data['confidence']}"
                if self.face_tracker.is_processed_for_unknown(face_id):
                    stability_text += " [P]"  # Processed marker
                
                # Get text size for background
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                (stab_width, stab_height), _ = cv2.getTextSize(stability_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                
                # Draw label background
                cv2.rectangle(frame, (x, y-30), (x + max(text_width, stab_width) + 10, y), color, cv2.FILLED)
                
                # Draw text
                cv2.putText(frame, label, (x+5, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, stability_text, (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Show system status
            status_y = 30
            cv2.putText(frame, f"Known faces: {len(self.known_face_names)}", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Processed unknown: {len(self.face_tracker.processed_unknown_faces)}", 
                       (10, status_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            
            # Display frame
            cv2.imshow('Stabilized Face Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Cleanup
        self.video_capture.release()
        cv2.destroyAllWindows()

def main():
    try:
        face_system = FaceRecognitionSystem()
        face_system.run_recognition()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed: pip install opencv-python deepface tensorflow scikit-learn")

if __name__ == "__main__":
    main()