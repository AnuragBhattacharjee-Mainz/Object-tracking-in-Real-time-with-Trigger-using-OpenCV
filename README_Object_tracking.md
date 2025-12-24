
# Object-tracking-in-Real-time-with-Trigger-using-OpenCV

_Building a custom model using Computer Vision that uses a camera feed to track the position of the user’s hand in Real time and detect when the hand approaches a virtual object on the screen. When the hand reaches the boundary, the system should trigger a clear on-screen warning: SAFE, WARNING OR DANGER, based on visual state feedback._

---

## 📌 Table of Contents
- <a href="#overview">Overview</a>
- <a href="#Object-tracking-problem">Object tracking Problem</a>
- <a href="#tools--technologies">Tools & Technologies</a>
- <a href="#project-structure">Project Structure</a>
- <a href="#how-to-run-this-project">How to Run This Project</a>
- <a href="#final-recommendations">Final Recommendations</a>
- <a href="#author--contact">Author & Contact</a>

---
<h2><a class="anchor" id="overview"></a>Overview</h2>

The main objective of this project is to build a prototype that utilises system camera to track the position of the user’s hand in real time and detect when the hand approaches a defined virtual object on the screen. Real-time performance Target ≥ 8 FPS on CPU-only execution. Libraries allowed: OpenCV and Numpy. The trigger warning logics should be clearly established.

---
<h2><a class="anchor" id="Object-tracking-problem"></a>Object tracking Problem</h2>

Object tracking is critical in the logistics, automobile, retail, medical-imaging and sports industry. This project aims to:
- Track the position of user's hand with precision in real time
- Trigger a custom message on the screen based on contour detection
- Selectively ignore bigger objects and utilize dynamic distance based state logic

---

<h2><a class="anchor" id="tools--technologies"></a>Tools & Technologies</h2>

- OpenCV Library
- Python (Numpy)
- GitHub

---
<h2><a class="anchor" id="project-structure"></a>Project Structure</h2>

```
Object-tracking-in-Real-time-with-Trigger-using-OpenCV/
│
├── README.md
├── requirements.txt
├── object_tracking_prototype_model.py
```

---
<h2><a class="anchor" id="how-to-run-this-project"></a>How to Run This Project</h2>

1. Clone the repository:
```bash
git clone https://github.com/AnuragBhattacharjee-Mainz/Object-tracking-in-Real-time-with-Trigger-using-OpenCV
```
2. Open vs code shell/terminal command and run
```bash
pip install opencv-python numpy
```
If pip install is not working properly, then first try: "python -m pip install --upgrade pip setuptools wheel"
and then try to install numpy and opencv.
If we want to execute it in a virtual environment the follow the code below:
```bash
python -m venv cv-env
cv-env\Scripts\activate
pip install --upgrade pip
pip install numpy opencv-python
```

3. Run the .py file in vs code based on system python version
```bash
python object_tracking_prototype_model.py
```
4. A New Window will then open and the messages will appear on the screen as per user's hand movement. 
Type "deactivate" to finally exit the python virtual environment.

---
<h2><a class="anchor" id="final-recommendations"></a>Final Recommendations</h2>

- The kernel size for GaussianBlur in opencv must be a odd number. 
- The CPU usuge in this project is depended on screen resolution and can be changed explicitely.
- Skin segmentation and detection can be further improved by reducing or tweaking Noise in create_skin_mask function.
- The distance for trigger is calculated from any part of the hand (contour) but it can be customised to centroid of the hand.
- The safe distance and warning distance can be adjusted accordingly by changing distance pixel.
- The face of the user is ignored in this prototype by the camera as it covers roughly 40-50% at the centre of the image .
- The prototype also ignores any object contours that is too far from the camera using a GATE, controls maximum hand area ratio
  and maximum allowed distance.

---
<h2><a class="anchor" id="author--contact"></a>Author & Contact</h2>

**Anurag Bhattacharjee**   
- 🔗 Data Science Practitioner
- 🔗 [LinkedIn](www.linkedin.com/in/anurag-process-analyst)
