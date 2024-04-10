import os
from deepface import DeepFace
import matplotlib.pyplot as plt

def find_images_with_highest_emotion_scores(directory_path):
    # Get a list of image files in the directory
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    if not image_files:
        return None  # No image files found in the directory

    # Create dictionaries to store the highest scores and their corresponding image paths for each emotion
    highest_scores = {
        "angry": -float('inf'),
        "disgust": -float('inf'),
        "fear": -float('inf'),
        "happy": -float('inf'),
        "sad": -float('inf'),
        "surprise": -float('inf'),
        "neutral": -float('inf')
    }
    highest_score_images = {
        "angry": None,
        "disgust": None,
        "fear": None,
        "happy": None,
        "sad": None,
        "surprise": None,
        "neutral": None
    }

    # Loop through each image in the directory
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)

        # Analyze the image for emotions using DeepFace
        try:
            #result = DeepFace.analyze(image_path, emotions=True)
            result = DeepFace.analyze(image_path, actions = ['emotion'], enforce_detection=False)

            # Update the highest scores and their corresponding image paths
            for emotion in highest_scores.keys():
                emotion_score = result[0]['emotion'][emotion]
                if emotion_score > highest_scores[emotion]:
                    highest_scores[emotion] = emotion_score
                    highest_score_images[emotion] = image_path
        except Exception as e:
            print(f"Error analyzing {image_file}: {str(e)}")

    return highest_score_images

# Example usage:
directory_path = "/content/gdrive/MyDrive/EmotionDetector/same_person_dump/all_pictures 2"
highest_score_images = find_images_with_highest_emotion_scores(directory_path)

# Plot the images with the highest scores for each emotion
for emotion, image_path in highest_score_images.items():
    if image_path:
        image = plt.imread(image_path)
        plt.figure(figsize=(2, 2))
        plt.title(f"Highest {emotion.capitalize()} Emotion Score")

        
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    else:
        print(f"No suitable images found for {emotion.capitalize()} emotion.")
