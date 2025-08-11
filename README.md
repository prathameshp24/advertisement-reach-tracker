Alright — here’s the updated **README** with an ASCII pipeline flow diagram inserted in the right spot so it visually explains how your project flows from CCTV video to database storage.

---

# 📹 Advertisement Reach & Potential Buyer Detection via CCTV

This project leverages **computer vision**, **deep learning models**, and **PostgreSQL** to automatically track advertisement reach and identify potential buyers from **CCTV feeds**.
It detects people, tracks gaze direction to see if they are focusing on the ad, classifies gender & age group, and stores results in a database with date-wise processed frames for auditing.

---

## 🚀 Features

* **People Detection** – Uses YOLOv8 for real-time person identification.
* **Face Detection** – Uses YOLOv8-face for accurate face localization.
* **Demographics Analysis** – Predicts **gender** and **age group** using pretrained Caffe models.
* **Gaze Tracking** – Determines if a person is looking at the camera/ad for a sustained duration.
* **Head Pose Estimation** – Uses Dlib facial landmarks & OpenCV’s `solvePnP` for precise yaw/pitch/roll angles.
* **Database Integration** – Stores summary & detailed records in **PostgreSQL** tables.
* **Performance Monitoring** – Logs operation-wise time consumption for profiling.
* **Date-wise Frame Archiving** – Saves processed frames for audit and review.
* **Linux-friendly** – Designed and tested in a basic Linux-based environment.

---

## 📂 Project Structure

```
├── predict11.py       # Main detection pipeline
├── monitor.py         # Manages DB creation, scheduling, and video processing
├── gender_net.caffemodel
├── gender_deploy.prototxt
├── age_net.caffemodel
├── age_deploy.prototxt
├── shape_predictor_68_face_landmarks.dat
└── processed_frames/  # Date-wise saved frames (auto-created)
```

---

## ⚙️ How It Works

```
┌─────────────────────┐
│   CCTV Feed / Video │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   monitor.py        │
│  - Creates DB       │
│  - Calls processing │
│  - Manages folders  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   predict11.py      │
│  1. Load models     │
│  2. Detect persons  │
│  3. Detect faces    │
│  4. Classify gender │
│     & age group     │
│  5. Track gaze      │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────────┐
│ Annotate frame & store image │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Insert into PostgreSQL DB    │
│  - Summary Table             │
│  - Details Table             │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Print performance summary    │
│  & processing times          │
└──────────────────────────────┘
```

---

## 🧠 Models & Libraries

| Library / Model                                                            | Purpose                                    |
| -------------------------------------------------------------------------- | ------------------------------------------ |
| **Ultralytics YOLOv8 (`yolov8n.pt`)**                                      | Fast person detection                      |
| **YOLOv8-face (`yolov8n-face.pt`)**                                        | High-accuracy face detection               |
| **Caffe Gender Model** (`gender_net.caffemodel`, `gender_deploy.prototxt`) | Gender classification                      |
| **Caffe Age Model** (`age_net.caffemodel`, `age_deploy.prototxt`)          | Age group classification                   |
| **Dlib 68 Landmark Model** (`shape_predictor_68_face_landmarks.dat`)       | Facial landmark detection                  |
| **OpenCV**                                                                 | Image preprocessing, head pose estimation  |
| **DeepFace**                                                               | (Optional) For emotion/sentiment detection |
| **Psycopg2**                                                               | PostgreSQL database operations             |
| **NumPy**                                                                  | Numerical computations                     |
| **Deque, OrderedDict**                                                     | Efficient tracking of faces over time      |

---

## 🛠 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ad-reach-tracker.git
cd ad-reach-tracker

# Install dependencies
pip install ultralytics opencv-python-headless numpy dlib deepface psycopg2-binary

# Download required model files
# (Ensure you have the .caffemodel, .prototxt, and .dat files in project root)
```

---

## ▶️ Usage

### 1️⃣ Run Monitoring Script

```bash
python monitor.py
```

### 2️⃣ Process a Single Video (manual mode)

```bash
python predict11.py /path/to/video.mp4
```

---

## 📊 Performance Monitoring

This project tracks:

* **Person Detection time**
* **Face Detection time**
* **Database operations time**
* **Total frame processing time**

A summary table is printed after processing:

```
Operation                     Total(s)   Avg(ms)    Min(ms)    Max(ms)    Calls
--------------------------------------------------------------------------------
Person Detection                12.35     45.32      40.21      60.45     273
Face Detection                   8.90     33.15      29.45      49.12     273
Database Operations              4.21     15.43      12.10      23.50     100
Total Frame Processing          26.00     95.00      85.00     110.00     273
================================================================================
```

---

## 🖼 Demo Video

📌 
[![Watch Demo](https://drive.google.com/file/d/13RYlB8OsnLWfLrYStzF8euBJvYVY8mKn/view?usp=sharing)

---

## 🖥 Process-Time Screenshot (Linux Environment)

📌 
![Performance Screenshot](https://drive.google.com/file/d/1ih5PfmxsBmPkVcs1rr2msTv7ODXOzy1E/view?usp=sharing)

---

## 📌 Database Schema

**`image_proc_summary`**

| Column               | Description                   |
| -------------------- | ----------------------------- |
| device\_id           | CCTV device identifier        |
| feed\_time           | Original feed timestamp       |
| process\_time\_stamp | Processing timestamp          |
| people\_count        | Total people detected         |
| male\_count          | Males detected                |
| female\_count        | Females detected              |
| image\_path          | Path to processed frame image |

**`image_proc_details`**

| Column     | Description                       |
| ---------- | --------------------------------- |
| person\_id | Internal tracking ID              |
| gazing     | Whether person was gazing at ad   |
| gender     | Detected gender                   |
| age\_group | Age group classification          |
| sentiment  | Default 'normal', can be extended |

---

## 📌 Audit Logging

* Every processed frame is saved under:

  ```
  processed_frames/YYYY-MM-DD/video_name/img_proc_id.jpg
  ```
* Ensures traceability for advertisement reach validation.

---

## 📜 License

This project is licensed under the MIT License – you’re free to use and modify with attribution.

---
