# AI-Based Frictionless People Counting and Tracking System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 2.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NVIDIA Jetson](https://img.shields.io/badge/Hardware-NVIDIA%20Jetson%20AGX-green.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/)

An intelligent, real-time people counting and tracking system developed for **Engex 2025**, utilizing computer vision and deep learning to accurately track individuals across multiple zones using face detection and unique ID assignment. The system provides contactless, automated monitoring without physical barriers.

## 🎯 Overview

This system addresses the major drawback of processing delays in conventional people detection and tracking systems by implementing a **vision-based real-time** solution using off-the-shelf IP cameras with **high accuracy**. The system detects faces, assigns unique identifiers, and tracks movement between zones (entrance and exit) with real-time visualization through a web interface.

## ✨ Key Features

- **Real-time Face Detection**: YOLOv8-powered face detection for instant recognition
- **Unique ID Assignment**: Each detected face receives a unique identifier upon entry
- **Bi-directional Tracking**: Separate tracking for entries and exits across zones
- **Two-Zone Implementation**: Monitors Zone 1 and Zone 2 with independent counting
- **Centroid Tracking**: Advanced tracking method with threshold logic to avoid repeat counts
- **Real-time Web Dashboard**: Live visualization showing zone occupancy and people distribution
- **RTSP Camera Support**: Seamless integration with IP cameras via RTSP streams
- **Edge Computing**: Runs on NVIDIA Jetson AGX for efficient on-device processing
- **Privacy-Focused**: No personal data storage, only anonymous tracking IDs

## 🏗️ System Architecture

### Implementation for Two Zones

```
┌─────────────────────────────────────────────────────────┐
│                    Camera Setup                          │
│  Zone 1: Entrance Camera + Exit Camera                   │
│  Zone 2: Entrance Camera + Exit Camera                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Video Stream Input (RTSP)                   │
│              GStreamer Pipeline                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              YOLOv8 Face Detection                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Centroid Tracking with Threshold Logic           │
│         (Avoid Repeat Counts)                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│    Unique ID Assignment at Entrance/Exit Detection       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Zone Count Calculation (Zone 1 ↔ Zone 2)               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Real-Time Web Interface Dashboard                │
│    (Displays: Zone 1 Count + Zone 2 Count + Total)      │
└─────────────────────────────────────────────────────────┘
```

## 🔄 Operation Flow

1. **Face Detection**: System detects faces in the camera feed using YOLOv8
2. **ID Assignment**: Each face is assigned a unique tracking number at entrance/exit points
3. **Zone Tracking**: System tracks movement between Zone 1 and Zone 2
4. **Count Update**: Real-time counter updates displayed on web interface
5. **Visualization**: Dashboard shows zone occupancy with visual representation

## 🛠️ System Technical Information

### Hardware
- **Processing Unit**: NVIDIA Jetson AGX
- **Cameras**: Two IP cameras per zone (4 cameras total)
  - Entrance camera for Zone 1
  - Exit camera for Zone 1
  - Entrance camera for Zone 2
  - Exit camera for Zone 2

### Software Stack
- **Detection Model**: YOLOv8 for face detection
- **Tracking Method**: Centroid tracking with threshold logic to avoid repeat counts
- **Video Pipeline**: GStreamer for efficient video stream handling
- **Network Protocol**: RTSP links through IP addresses
- **Programming Language**: Python 3.8+
- **Deep Learning Framework**: PyTorch/TensorFlow
- **Computer Vision**: OpenCV
- **Web Framework**: Flask/FastAPI for dashboard
- **Real-time Processing**: Multi-threaded architecture

## 📋 Prerequisites

- NVIDIA Jetson AGX (or compatible NVIDIA device with CUDA support)
- Python 2.8+
- IP cameras with RTSP support
- Network connectivity for camera streams
- CUDA-enabled environment

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Irushi-coder/AI-Based-Frictionless-People-Counting-System-Engex_2025.git
cd AI-Based-Frictionless-People-Counting-System-Engex_2025
```

### 2. Set Up Python Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install System Dependencies (Jetson)
```bash
# GStreamer installation
sudo apt-get update
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# Additional dependencies for Jetson
sudo apt-get install python3-opencv
```

### 5. Configure Camera Settings
```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your RTSP camera URLs
```

### 6. Download YOLOv8 Model
```bash
python scripts/download_yolov8_model.py
```

## 💻 Usage

### Configure RTSP Camera Streams

Edit `config.yaml`:
```yaml
cameras:
  zone1:
    entrance: "rtsp://192.168.1.101:554/stream1"
    exit: "rtsp://192.168.1.102:554/stream1"
  zone2:
    entrance: "rtsp://192.168.1.103:554/stream1"
    exit: "rtsp://192.168.1.104:554/stream1"

detection:
  model: "yolov8n-face.pt"
  confidence: 0.6
  
tracking:
  centroid_threshold: 50  # pixels
  max_disappeared: 30  # frames
```

### Run the System

```bash
# Start the tracking system
python main.py

# Launch web dashboard (in separate terminal)
python dashboard.py
```

### Access the Dashboard
Open your browser and navigate to:
```
http://localhost:5000
```
or
```
http://<jetson-ip-address>:5000
```

## 📊 Applications

### Smart Buildings
- Automated occupancy monitoring
- Real-time space utilization tracking
- Energy management based on occupancy

### Workplaces & Universities
- Attendance and access management
- Room occupancy tracking
- Campus flow analysis

### Event Venues
- Crowd monitoring and safety control
- Zone capacity management
- Entry/exit flow optimization

### Factories & Logistics
- Worker flow tracking
- Zone monitoring for safety compliance
- Production area access control

## 📁 Project Structure

```
AI-Based-Frictionless-People-Counting-System-Engex_2025/
├── main.py                      # Main application entry
├── dashboard.py                 # Web dashboard server
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
├── README.md                    # Documentation
│
├── models/
│   └── yolov8n-face.pt         # Face detection model
│
├── src/
│   ├── detector.py             # YOLOv8 face detection
│   ├── tracker.py              # Centroid tracking logic
│   ├── counter.py              # Zone counting algorithm
│   ├── stream_handler.py       # GStreamer RTSP handling
│   └── utils.py                # Utility functions
│
├── dashboard/
│   ├── app.py                  # Flask/FastAPI app
│   ├── templates/
│   │   └── index.html          # Dashboard UI
│   └── static/
│       ├── css/
│       ├── js/
│       └── images/
│
├── data/
│   ├── logs/                   # System logs
│   └── counts.db               # Count database
│
└── scripts/
    ├── download_yolov8_model.py
    └── test_camera_connection.py
```

## ⚙️ Performance Metrics

- **Detection Speed**: 30+ FPS on NVIDIA Jetson AGX
- **Face Detection Accuracy**: 95%+ in controlled lighting
- **Tracking Accuracy**: 92-98% with proper camera placement
- **Latency**: < 100ms per frame processing
- **Multi-stream Processing**: Handles 4 simultaneous RTSP streams

## 🎓 Technical Highlights

### Centroid Tracking Algorithm
- Calculates center point of detected faces
- Maintains unique IDs across frames
- Threshold logic prevents duplicate counting
- Handles temporary occlusions (max_disappeared parameter)

### Zone Transition Logic
```
Entry Detection → Assign Unique ID → Track across frames → 
Zone 1 → Zone 2 (Count +1 in Zone 2, -1 in Zone 1) →
Exit Detection → Remove ID
```

## 👥 Team & Collaboration

**Developed in collaboration with:**
- **Zone24x7** - Industry Partner
- **University of Peradeniya** - Faculty Of Engineering
- **EngEx** - Engineering Exhibition Platform

**Contact Information:**
- Supervisor:Dr.Isuru Dassanayake - isurud@ee.pdn.ac.lk
- Team: U.K.I.Layanga - e19217@eng.pdn.ac.lk
        S.W.C.Sandaruwan - e19350@eng.pdn.ac.lk
        M.W.S.Lakmina - e19212@eng.pdn.ac.lk

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics YOLOv8** - Face detection model
- **NVIDIA** - Jetson AGX platform and CUDA support
- **OpenCV & GStreamer** - Video processing libraries
- **Zone24x7** - Industry collaboration and support
- **University of Peradeniya** - Academic guidance
- **EngEx 2025** - Platform and opportunity

## 🔮 Future Enhancements

- [ ] Cloud integration for remote monitoring
- [ ] Mobile app for iOS and Android
- [ ] Advanced analytics with heat maps
- [ ] Multi-building support
- [ ] Integration with existing access control systems
- [ ] Machine learning for behavior pattern analysis
- [ ] API for third-party integrations
- [ ] Support for additional zones (scalable architecture)

**Developed for Engex 2025** | University of Peradeniya | In Collaboration with Zone24x7

*Engineering Innovation for Intelligent Space Management*
