from re import split
from flask import Flask,render_template,request 
from flask_bootstrap import Bootstrap
import argparse
import camera_demo
from camera_demo import main
import camera_demo
from deep_emotion_recognition import DeepEmotionRecognizer
import os
import pyaudio
import wave



app = Flask(__name__)
Bootstrap(app)
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/home')
def inde():
    return render_template("index.html")

@app.route('/qstn')
def phq():
    return render_template("phq9.html",data="تشخیص احساسات ماثر در افسردگی با صدا و تصویر")

@app.route('/expression') 
def expression():
    camera_demo=main()
    if __name__ == '__main__':   
        
        P=app.run(debug=True)
    return function("main",data = P)




@app.route('/face') 
def face():
    return render_template("face.html",data = "تشخیص احساسات از تصویر")

@app.route('/voice')
def voice():
    return render_template("voice.html",data = "برای ضبط صدا روی میکروفون کلیک کنید")

@app.route('/voice_recording')
def voice_recording():
    CHUNK = 1024 
    FORMAT = pyaudio.paInt16 #paInt8
    CHANNELS = 2 
    RATE = 44100 #sample rate
    RECORD_SECONDS = 4
    WAVE_OUTPUT_FILENAME = "DepressionProjectOutput.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK) #buffer

    #return render_template("voice.html", data = "Recording ....")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data) # 2 bytes(16 bits) per channel

    

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return render_template("voice.html", data = "ضبط تمام شد")

@app.route('/voice_analyzer')
def voice_analyzeer():

    return render_template("voice.html",data ="")
        




    