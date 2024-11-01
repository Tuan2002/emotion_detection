# webcam/views.py
import cv2
from django.shortcuts import render
import numpy as np
from django.http import JsonResponse
from fer import FER
from PIL import Image
import base64
from io import BytesIO

emotion_detector = FER()

emotion_translations = {
    "happy": "Vui vẻ",
    "sad": "Buồn bã",
    "angry": "Tức giận",
    "surprised": "Ngạc nhiên",
    "fearful": "Sợ hãi",
    "disgusted": "Ghê tởm",
    "neutral": "Bình thường",
    "confused": "Bối rối",
    "excited": "Hào hứng",
    "bored": "Chán nản"
}

def detect_emotion(request):
    if request.method == "POST":
        image_data = request.POST.get("image")
        image_data = base64.b64decode(image_data.split(",")[1])
        image = Image.open(BytesIO(image_data))
        bgr_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Detect emotions in the image
        emotions = emotion_detector.detect_emotions(bgr_image)
        if emotions:
            # Get the dominant emotion, the emotion with the highest probability
            dominant_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
            vietnamese_emotion = emotion_translations.get(dominant_emotion, "Không xác định")
            return JsonResponse({"emotion": vietnamese_emotion})
        else:
            return JsonResponse({"emotion": "Không thể nhận dạng khuôn mặt"})
    return JsonResponse({"error": "Invalid request"}, status=400)

def index(request):
    return render(request, 'webcam/index.html')
