from deep_emotion_recognition import DeepEmotionRecognizer
import pyaudio
import os
import wave
from sys import byteorder
from array import array
from struct import pack
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier

from utilsspeech import get_best_estimators

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 30

def is_silent(snd_data):
    "اگر  «بی‌صدا» باشد، «درست» را برمی‌گرداند"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "میزان و ولوم صدا را میانگین میگیرد"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "بی صدایی آخر و اول صدا را برش می دهد"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # برش سمت چپ
    snd_data = _trim(snd_data)

    # برش سمت راست
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    " ثانیه اول و آخر را بدون صدا می گذارد"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
ضبط یک یا چند کلمه و عادی سازی  صوت
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "ضبط با میکروفون با نام و مسیر تعریف شده"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def get_estimators_name(estimators):
    result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}



if __name__ == "__main__":
    estimators = get_best_estimators(True)
    estimators_str, estimator_dict = get_estimators_name(estimators)
    import argparse
    parser = argparse.ArgumentParser(description="""
                                    Testing emotion recognition system using your voice, 
                                    please consider changing the model and/or parameters as you wish.
                                    """)
    parser.add_argument("-e", "--emotions", help=
                                            """Emotions to recognize separated by a comma ',', available emotions are
                                            "neutral", "calm", "happy" "sad", "angry", "fear", "disgust", "ps" (pleasant surprise)
                                            and "boredom", default is "sad,neutral,happy"
                                            """, default="sad,neutral,happy")
    parser.add_argument("-m", "--model", help=
                                        """
                                        The model to use, 8 models available are: {},
                                        default is "BaggingClassifier"
                                        """.format(estimators_str), default="BaggingClassifier")



    args = parser.parse_args()

    features = ["mfcc", "chroma", "mel"]
    deeprec = DeepEmotionRecognizer(emotions=['angry', 'Depressed', 'neutral', 'surprised', 'happy'], n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)
    deeprec.train()
    print(deeprec.test_score())  
    print("Please talk")

    filename = "Output.wav"
    record_to_file(filename)
    prediction = deeprec.predict('Output.wav')
    print(f"Prediction: {prediction}")
    print(deeprec.predict_proba("Output.wav"))