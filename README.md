# Human Detection and Tracking Drone

This project implements a human detection and tracking system using a drone's video feed. It leverages state-of-the-art object detection models from Hugging Face and OpenCV's tracking algorithms to detect and track humans in real-time.

## Features

- Real-time human detection using the `facebook/detr-resnet-50` model from Hugging Face Transformers.
- Multi-object tracking with OpenCV trackers.
- Motion trail visualization for tracked humans.
- Compatible with webcam or drone video feed input.

## Requirements

- Python 3.6 or higher
- OpenCV
- NumPy
- Pillow
- Transformers (Hugging Face)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows (cmd)
venv\Scripts\activate
# On Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script to start human detection and tracking:

```bash
python human_detection_tracking_drone.py
```

Press `ESC` to exit the application.

## How It Works

- The script captures video frames from the webcam.
- Every 24 frames, it runs the Hugging Face object detection pipeline to detect humans.
- It initializes or updates OpenCV trackers for detected humans.
- Motion trails are drawn to visualize movement paths.
- Bounding boxes and labels are displayed on the video feed.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [OpenCV](https://opencv.org/)
