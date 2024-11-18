import numpy as np
import cv2
from collections import Counter
import webcolors

def closest_color(requested_color):
    min_colors = {}
    for name in webcolors.names("css3"):
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_brightness_category(average_brightness):
    if average_brightness < 50:
        return "Dark"
    elif 50 <= average_brightness < 100:
        return "Dim"
    elif 100 <= average_brightness < 150:
        return "Neutral"
    elif 150 <= average_brightness < 200:
        return "Bright"
    else:
        return "Very Bright"

def calculate_saturation(rgb):
    r, g, b = rgb
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    # Saturation as difference between max and min values normalized by max value
    return (max_val - min_val) / max_val if max_val != 0 else 0

def classify_color(rgb):
    r, g, b = rgb
    saturation = calculate_saturation(rgb)
    if saturation < 0.2:
        return "neutral"  # Low saturation colors are likely gray or other neutrals
    elif r > g and r > b:
        return "warm"  # Red-dominant colors are warm
    elif b > r and b > g:
        return "cool"  # Blue-dominant colors are cool
    elif abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30:
        return "neutral"  # Close RGB values suggest a neutral color (like gray)
    else:
        return "cool" if b > r else "warm"

def describe_mood(brightness_category, dominant_colors):
    warm_colors = sum(1 for color in dominant_colors if classify_color(color) == "warm")
    cool_colors = sum(1 for color in dominant_colors if classify_color(color) == "cool")
    neutral_colors = sum(1 for color in dominant_colors if classify_color(color) == "neutral")

    # Determine dominant color tone
    if warm_colors > cool_colors and warm_colors > neutral_colors:
        color_tone = "warm"
    elif cool_colors > warm_colors and cool_colors > neutral_colors:
        color_tone = "cool"
    else:
        color_tone = "neutral"

    # Map brightness and color tone to mood descriptions
    if brightness_category == "Dark":
        if color_tone == "cool":
            return "Mysterious and Tense"
        elif color_tone == "warm":
            return "Cozy and Intimate"
        else:
            return "Dark and Neutral, perhaps Melancholy"
    elif brightness_category == "Dim":
        if color_tone == "cool":
            return "Gloomy and Reflective"
        elif color_tone == "warm":
            return "Warm and Nostalgic"
        else:
            return "Muted and Serious"
    elif brightness_category == "Neutral":
        if color_tone == "warm":
            return "Calm and Inviting"
        elif color_tone == "cool":
            return "Balanced and Reflective"
        else:
            return "Relaxed and Natural"
    elif brightness_category == "Bright":
        if color_tone == "warm":
            return "Joyful and Energetic"
        elif color_tone == "cool":
            return "Bright and Peaceful"
        else:
            return "Fresh and Neutral"
    else:  # "Very Bright"
        if color_tone == "warm":
            return "Vibrant and Exciting"
        elif color_tone == "cool":
            return "Open and Serene"
        else:
            return "Bright and Clean"

def analyze_lighting_and_color(video_path):
    print("analyzing lighting and colors in scene... ")
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    brightness_values = []
    all_colors = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to HSV (Hue, Saturation, Value) to assess brightness and color
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness_values.append(np.mean(hsv_frame[:, :, 2]))  # Value channel for brightness
        # Resize frame for faster color processing
        resized_frame = cv2.resize(frame, (50, 50))  # Resize to speed up analysis
        reshaped_frame = resized_frame.reshape(-1, 3)
        # Convert each BGR color to a named color
        for pixel in reshaped_frame:
            all_colors.append((pixel[2], pixel[1], pixel[0]))
    cap.release()

    # Calculate average brightness
    avg_brightness = np.mean(brightness_values)
    brightness_category = get_brightness_category(avg_brightness)

    # Count most common colors
    color_counts = Counter(all_colors).most_common(10)  
    rgb_colors = set()
    color_names = set()
    for color, count in color_counts:
        closest = closest_color(color)
        rgb_colors.add(color)
        color_names.add(closest)

    print("bright: ", brightness_category)
    print("colors: ", color_names)
    mood = describe_mood(brightness_category, rgb_colors)
    print(f"Finished processing lighting and color for scene {video_path}")
    return mood
  