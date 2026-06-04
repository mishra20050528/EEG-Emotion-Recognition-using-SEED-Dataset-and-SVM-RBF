
print("\n")
print("=" * 60)
print("EMOTION BASED RECOMMENDATION SYSTEM")
print("=" * 60)


unique, counts = np.unique(y_pred, return_counts=True)

emotion_count = dict(zip(unique, counts))

dominant_class = max(emotion_count, key=emotion_count.get)

emotion_names = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

dominant_emotion = emotion_names[dominant_class]

print(f"\nDominant Emotion Detected: {dominant_emotion}")

music_recommendations = {
    "Negative": [
        "Weightless - Marconi Union",
        "River Flows In You - Yiruma",
        "Relaxing Piano Collection",
        "Calm Meditation Music",
        "Lo-Fi Focus Playlist"
    ],

    "Neutral": [
        "Perfect - Ed Sheeran",
        "Photograph - Ed Sheeran",
        "Someone You Loved",
        "Counting Stars",
        "A Thousand Years"
    ],

    "Positive": [
        "Happy - Pharrell Williams",
        "Believer - Imagine Dragons",
        "On Top of the World",
        "Can't Stop The Feeling",
        "Thunder"
    ]
}


movie_recommendations = {
    "Negative": [
        "3 Idiots",
        "The Intern",
        "Jumanji",
        "The Secret Life of Walter Mitty",
        "Forrest Gump"
    ],

    "Neutral": [
        "The Pursuit of Happyness",
        "The Social Network",
        "Hidden Figures",
        "The Martian",
        "A Beautiful Mind"
    ],

    "Positive": [
        "Avengers Endgame",
        "Top Gun Maverick",
        "John Wick",
        "Mission Impossible",
        "Mad Max Fury Road"
    ]
}


wellness_recommendations = {
    "Negative": [
        "Take a 10-minute break",
        "Practice deep breathing",
        "Listen to calming music",
        "Go for a short walk",
        "Try a mindfulness exercise"
    ],

    "Neutral": [
        "Continue current activity",
        "Maintain focus",
        "Take small hydration breaks",
        "Review current goals",
        "Stay consistent"
    ],

    "Positive": [
        "Attempt challenging tasks",
        "Learn a new concept",
        "Start a creative activity",
        "Work on high-priority goals",
        "Take advantage of high motivation"
    ]
}


print("\nRecommended Music:")
for song in music_recommendations[dominant_emotion]:
    print("  •", song)

print("\nRecommended Movies:")
for movie in movie_recommendations[dominant_emotion]:
    print("  •", movie)

print("\nWellness Suggestions:")
for item in wellness_recommendations[dominant_emotion]:
    print("  •", item)

print("\n" + "=" * 60)
print("RECOMMENDATION GENERATED SUCCESSFULLY")
print("=" * 60)
