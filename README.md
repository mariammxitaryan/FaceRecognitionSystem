Думал на протяжении секунды


````markdown
# FaceTool: Command‑Line Face Recognition & Analysis

A simple, extensible Python CLI for face recognition, attribute analysis, and face verification using the [DeepFace](https://github.com/serengil/deepface) library.

---

## 🔍 Features

- **Recognize**: Find the closest matches for a query face in a local “database” folder of face images.  
- **Analyze**: Estimate age, gender, emotion, and race composition of a detected face.  
- **Verify**: Compare two face images and decide whether they belong to the same person.

Every operation uses DeepFace’s high‑quality pretrained models (VGG‑Face, Facenet, ArcFace, etc.) and lets you choose different face‑detector backends (MTCNN, OpenCV, dlib, RetinaFace).

---

## 💾 Installation

1. **Clone the repo**  

2. **Create & activate** a virtual environment (recommended)

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install deepface pandas tensorflow scipy opencv-python mtcnn dlib
   ```

---

## 🚀 Usage

All commands are in the main script `face_tool.py`. Use the `--help` flag to see available options:

```bash
python face_tool.py --help
```

### 1. Recognize

Search a folder of known faces for matches to a query image:

```bash
python face_tool.py recognize \
  --img path/to/query.jpg \
  --db path/to/face_database/ \
  --model VGG-Face \
  --backend opencv
```

* **--img**: Path to the query image containing one face.
* **--db**: Directory with one image per known person.
* **--model**: Embedding model (`VGG-Face`, `Facenet`, `ArcFace`, …).
* **--backend**: Face detector (`opencv`, `mtcnn`, `dlib`, `retinaface`).
* **--no-enforce**: (Optional) Don’t error if no face is found.

### 2. Analyze

Estimate demographic and emotional attributes:

```bash
python face_tool.py analyze \
  --img path/to/face.jpg \
  --actions age gender emotion race \
  --model Facenet \
  --backend mtcnn \
  --out results.json
```

* **--actions**: One or more of `age`, `gender`, `emotion`, `race`.
* **--out**: (Optional) Path to save a JSON report.

### 3. Verify

Compare two face images:

```bash
python face_tool.py verify \
  --img1 path/to/personA.jpg \
  --img2 path/to/personB.jpg \
  --model ArcFace \
  --metric cosine \
  --backend dlib
```

* **--metric**: Distance metric (`cosine`, `euclidean`, `euclidean_l2`).

---

## 📂 Project Structure

```
.
├── face_tool.py     # Main CLI script
├── README.md        # This file
└── tests/           # (Optional) unit tests and sample images
```

---

## 🧪 Testing

If you add a `tests/` folder with pytest tests, run:

```bash
pytest -q
```

---

## 💡 Extensions

* Add more subcommands (e.g. extract embeddings, batch‑process folders).
* Integrate with a web UI or desktop GUI.
* Support live webcam input.

---
