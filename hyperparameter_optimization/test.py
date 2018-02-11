from sklearn.metrics import make_scorer, fbeta_score

ftwo_scorer = make_scorer(fbeta_score, beta=2)