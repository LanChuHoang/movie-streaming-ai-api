from joblib import load

model_file_path = "./saved_models/movie_review_sa_v1.joblib"
model = load(model_file_path)
