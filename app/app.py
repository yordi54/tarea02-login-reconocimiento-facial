import cv2
from flask import Flask, render_template, Response, redirect, url_for

app = Flask(__name__, static_folder='static')
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:/Users/yordi/AppData/Local/Programs/Python/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if face_cascade.empty():
        raise Exception("No se pudo cargar el clasificador de cascada para la detección facial")
    return faces

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            faces = detect_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

def redirigir():
    return redirect(url_for('index.html'))

@app.route('/home')
def  home():
    return render_template('app.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=8000)