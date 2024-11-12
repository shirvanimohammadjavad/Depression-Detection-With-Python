import soundfile
import librosa
import numpy as np
import pickle
import os
from convert_wavs import convert_audio


AVAILABLE_EMOTIONS = {
    "neutral",#خنثی
    "calm",#آرام
    "happy",#شاد
    "sad",#غمگین
    "angry",#عصبانی
    "fear",#ترسیده یا مضطرب
    "disgust",#منزجر
    "ps", #شگفت زده (خوب)  pleasant surprised
    "boredom"#بی حوصله
}


def get_label(audio_config):
    """برچسب مربوط به ویژگی هایی که قرار است استخراج شوند را برمی گرداند
       به عنوان مثال
    audio_config = {'mfcc': True, 'chroma': True, 'contrast': False, 'tonnetz': False, 'mel': False}
    get_label(audio_config): 'mfcc-chroma'
    """
    features = ["mfcc", "chroma", "mel", "contrast", "tonnetz"]
    label = ""
    for feature in features:
        if audio_config[feature]:
            label += f"{feature}-"
    return label.rstrip("-")


def get_dropout_str(dropout, n_layers=3):
    if isinstance(dropout, list):
        return "_".join([ str(d) for d in dropout])
    elif isinstance(dropout, float):
        return "_".join([ str(dropout) for i in range(n_layers) ])


def get_first_letters(emotions):
    return "".join(sorted([ e[0].upper() for e in emotions ]))


def extract_feature(file_name, **kwargs):
    """
    استخراج ویژگی ها
        الگوریتم های ویژگی صوتی:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        به عنوان مثال
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    try:
        with soundfile.SoundFile(file_name) as sound_file:
            pass
    except RuntimeError:
        #اگر فرمت مناسب پردازش نباشد فرآیند تبدیل را انجام می دهد
        # دریافت نام پایه
        basename = os.path.basename(file_name)
        dirname  = os.path.dirname(file_name)
        name, ext = os.path.splitext(basename)
        new_basename = f"{name}_c.wav"
        new_filename = os.path.join(dirname, new_basename)
        v = convert_audio(file_name, new_filename)
        if v:
            raise NotImplementedError("Converting the audio files failed, make sure `ffmpeg` is installed in your machine and added to PATH.")
    else:
        new_filename = file_name
    with soundfile.SoundFile(new_filename) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result


def get_best_estimators(classification):
    """
بارگیری برآورد گر بهترین مدل
    """
    if classification:
        return pickle.load(open("grid/best_classifiers.pickle", "rb"))
    else:
        return pickle.load(open("grid/best_regressors.pickle", "rb"))


def get_audio_config(features_list):
    """
پیکربندی ویژگی ها
    """
    audio_config = {'mfcc': False, 'chroma': False, 'mel': False, 'contrast': False, 'tonnetz': False}
    for feature in features_list:
        if feature not in audio_config:
            raise TypeError(f"Feature passed: {feature} is not recognized.")
        audio_config[feature] = True
    return audio_config
    