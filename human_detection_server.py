import cv2
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify
import threading
import time

class HumanDetectionServer:
    def __init__(self):
        self.app = Flask(__name__)
        self.model = YOLO("yolov8n.pt")
        self.cap = cv2.VideoCapture(0)  # Changed to 0 for default camera
        self.is_running = False
        self.person_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0
        
        # Setup routes
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')
            
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
                          
        @self.app.route('/status')
        def status():
            return jsonify({
                'detections': self.person_count,
                'fps': self.fps,
                'is_running': self.is_running
            })
            
    def generate_frames(self):
        self.is_running = True
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Run detection
            results = self.model(frame)[0]
            self.person_count = 0
            
            for box in results.boxes:
                cls = int(box.cls[0])
                
                # Class 0 = Person in COCO dataset
                if cls == 0:
                    self.person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw red dot at center top
                    cv2.circle(frame, (cx, y1 + 10), 5, (0, 0, 255), -1)

                    # Add label
                    cv2.putText(frame, f'Human {self.person_count}', (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Display total count
            cv2.putText(frame, f'Total Humans: {self.person_count}', (20, 40),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Calculate FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_time = current_time
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
    def run(self):
        try:
            import socket
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            
            print(f"🚀 Human Detection Server Starting...")
            print(f"📱 For mobile access:")
            print(f"   Local: http://localhost:5000")
            print(f"   Network: http://{local_ip}:5000")
            print(f"   Make sure your phone is on the same WiFi network!")
            
            self.app.run(host='0.0.0.0', port=5000, debug=False)
            
        finally:
            self.is_running = False
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    server = HumanDetectionServer()
    server.run()

