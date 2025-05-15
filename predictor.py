def predict_admission(score, major):
    major = major.lower()
    threshold = {
        "cntt": 24,
        "kinh tế": 22,
        "ngôn ngữ": 20
    }.get(major, 21)

    return score >= threshold
