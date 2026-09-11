# Hand Filter Project

A simple computer vision project using OpenCV and MediaPipe to detect two hands and apply visual filters to the area between them.

## Features

* Two-hand tracking with MediaPipe
* Detects index fingers and thumbs
* Creates a rectangular area between the hands
* Applies different visual filters inside the area
* Changes the active filter when the hands move close together
* Supports multiple custom OpenCV filters

## Requirements

* Python 3.10+
* OpenCV
* MediaPipe
* NumPy

Install dependencies:

```bash
pip install opencv-python mediapipe numpy
```

## Setup

1. Clone the repository.
2. Place the MediaPipe hand landmark model in the project directory.
3. Make sure `MODEL_PATH` points to the model file.
4. Connect a webcam.
5. Run:

```bash
python main.py
```

## Controls

* Move both hands to create the filter area.
* Move the hands close together to change the filter.
* Press `Q` to quit.

## Project Structure

```text
project/
├── main.py
├── filters.py
├── hand_landmarker.task
└── README.md
```

## Filters

The project includes several filters such as:

* Grid
* Color mapping
* Glitch
* Sepia
* White blur
* Pink pixel effect
* Fire
* Neon
* Pixelation
* Invert
* Pop art

New filters can be added to `filters.py` and then included in the `FILTERS` list.

## License

This project is for educational and experimental purposes.
