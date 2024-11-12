import glob
import pandas as pd
import os


def write_emodb_csv(emotions=["sad", "neutral", "happy"], train_name="train_emo.csv",
                    test_name="test_emo.csv", train_size=0.8, verbose=1):
    """
مجموعه داده ی  
EmoDob
گفتار را از دایرکتوری می خواند 
و آن را در فایل جدول می نویسد.
    """
    target = {"path": [], "emotion": []}
    categories = {
        "W": "angry",
        "L": "boredom",
        "E": "disgust",
        "A": "fear",
        "F": "happy",
        "T": "sad",
        "N": "neutral"
    }
    # حذف احساسات نامشخص 
    categories_reversed = { v: k for k, v in categories.items() }
    for emotion, code in categories_reversed.items():
        if emotion not in emotions:
            del categories[code]
    for file in glob.glob("data/emodb/wav/*.wav"):
        try:
            emotion = categories[os.path.basename(file)[5]]
        except KeyError:
            continue
        target['emotion'].append(emotion)
        target['path'].append(file)
    if verbose:
        print("[EMO-DB] Total files to write:", len(target['path']))
        
    # تقسیم مجموعه های آموزشی/آزمایشی
    n_samples = len(target['path'])
    test_size = int((1-train_size) * n_samples)
    train_size = int(train_size * n_samples)
    if verbose:
        print("[EMO-DB] Training samples:", train_size)
        print("[EMO-DB] Testing samples:", test_size)   
    X_train = target['path'][:train_size]
    X_test = target['path'][train_size:]
    y_train = target['emotion'][:train_size]
    y_test = target['emotion'][train_size:]
    pd.DataFrame({"path": X_train, "emotion": y_train}).to_csv(train_name)
    pd.DataFrame({"path": X_test, "emotion": y_test}).to_csv(test_name)


def write_tess_ravdess_csv(emotions=["sad", "neutral", "happy"], train_name="train_tess_ravdess.csv",
                            test_name="test_tess_ravdess.csv", verbose=1):
    """
مجموعه داده های  
 TESS و RAVDESS 
گفتار را از دایرکتوری می خواند 
و آن را در فایل جدول می نویسد.
    """
    train_target = {"path": [], "emotion": []}
    test_target = {"path": [], "emotion": []}
    
    for category in emotions:
        # آموزش دایرکتوری گفتار
        total_files = glob.glob(f"data/training/Actor_*/*_{category}.wav")
        for i, path in enumerate(total_files):
            train_target["path"].append(path)
            train_target["emotion"].append(category)
        if verbose and total_files:
            print(f"[TESS&RAVDESS] There are {len(total_files)} training audio files for category:{category}")
    
        # برای اعتبار سنجی دسته بندی داده های گفتاری
        total_files = glob.glob(f"data/validation/Actor_*/*_{category}.wav")
        for i, path in enumerate(total_files):
            test_target["path"].append(path)
            test_target["emotion"].append(category)
        if verbose and total_files:
            print(f"[TESS&RAVDESS] There are {len(total_files)} testing audio files for category:{category}")
    pd.DataFrame(test_target).to_csv(test_name)
    pd.DataFrame(train_target).to_csv(train_name)


def write_custom_csv(emotions=['sad', 'neutral', 'happy'], train_name="train_custom.csv", test_name="test_custom.csv",
                    verbose=1):
    """
    داده های صوتی سفارشی را از  می خواند و سپس فایل های توضیحات را می نویسد
    (csv)
    """
    train_target = {"path": [], "emotion": []}
    test_target = {"path": [], "emotion": []}
    for category in emotions:
        #  آموزش مجموعه داده
        for i, file in enumerate(glob.glob(f"data/train-custom/*_{category}.wav")):
            train_target["path"].append(file)
            train_target["emotion"].append(category)
        if verbose:
            try:
                print(f"[Custom Dataset] There are {i} training audio files for category:{category}")
            except NameError:
                # in case {i} doesn't exist
                pass
        
        # آزمون مجموعه داده
        for i, file in enumerate(glob.glob(f"data/test-custom/*_{category}.wav")):
            test_target["path"].append(file)
            test_target["emotion"].append(category)
        if verbose:
            try:
                print(f"[Custom Dataset] There are {i} testing audio files for category:{category}")
            except NameError:
                pass
    
    # CSVنوشتن جدول 
    if train_target["path"]:
        pd.DataFrame(train_target).to_csv(train_name)

    if test_target["path"]:
        pd.DataFrame(test_target).to_csv(test_name)