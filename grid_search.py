"""
جست و جو شبکه ای برای پیدا کردن بهترین مدل و پارامتر های ارائه شده 
"""

import pickle

from emotion_recognition import EmotionRecognizer
from parameters import classification_grid_parameters, regression_grid_parameters


emotions = ['sad', 'neutral', 'happy']

n_jobs = 4

best_estimators = []

for model, params in classification_grid_parameters.items():
    if model.__class__.__name__ == "KNeighborsClassifier":

        params['n_neighbors'] = [len(emotions)]
    d = EmotionRecognizer(model, emotions=emotions)
    d.load_data()
    best_estimator, best_params, cv_best_score = d.grid_search(params=params, n_jobs=n_jobs)
    best_estimators.append((best_estimator, best_params, cv_best_score))
    print(f"{emotions} {best_estimator.__class__.__name__} achieved {cv_best_score:.3f} cross validation accuracy score!")

print(f"[+] Pickling best classifiers for {emotions}...")
pickle.dump(best_estimators, open(f"grid/best_classifiers.pickle", "wb"))

best_estimators = []

for model, params in regression_grid_parameters.items():
    if model.__class__.__name__ == "KNeighborsRegressor":

        params['n_neighbors'] = [len(emotions)]
    d = EmotionRecognizer(model, emotions=emotions, classification=False)
    d.load_data()
    best_estimator, best_params, cv_best_score = d.grid_search(params=params, n_jobs=n_jobs)
    best_estimators.append((best_estimator, best_params, cv_best_score))
    print(f"{emotions} {best_estimator.__class__.__name__} achieved {cv_best_score:.3f} cross validation MAE score!")

print(f"[+] Pickling best regressors for {emotions}...")
pickle.dump(best_estimators, open(f"grid/best_regressors.pickle", "wb"))
