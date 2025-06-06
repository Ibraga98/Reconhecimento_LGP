import cv2
import numpy as np
import json
import mediapipe as mp
from tensorflow.keras.models import load_model

# Parâmetros
IMG_SIZE = 100

# Carregar classes dinamicamente
with open("classes.json", "r") as f:
    class_indices = json.load(f)
CLASSES = [None] * len(class_indices)
for nome, idx in class_indices.items():
    CLASSES[idx] = nome

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Carregar modelo
modelo = load_model("modelo_gestos_lgp.h5")

# Ativar webcam
cap = cv2.VideoCapture(0)
print("📷 Webcam ativa. Pressiona 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Desenhar landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obter bounding box da mão
            h, w, _ = frame.shape
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            xmin, xmax = int(min(xs) * w), int(max(xs) * w)
            ymin, ymax = int(min(ys) * h), int(max(ys) * h)

            # Garantir margens
            xmin = max(xmin - 20, 0)
            ymin = max(ymin - 20, 0)
            xmax = min(xmax + 20, w)
            ymax = min(ymax + 20, h)

            # Cortar imagem da mão
            mao = frame[ymin:ymax, xmin:xmax]
            if mao.size == 0:
                continue

            # Pré-processamento
            mao = cv2.resize(mao, (IMG_SIZE, IMG_SIZE))
            mao = mao.astype("float32") / 255.0
            mao = np.expand_dims(mao, axis=0)

            # Predição
            pred = modelo.predict(mao, verbose=0)
            classe_idx = np.argmax(pred)
            confianca = np.max(pred)
            classe = CLASSES[classe_idx]

            # Mostrar predição
            texto = f"{classe.upper()} ({confianca*100:.1f}%)"
            cv2.putText(frame, texto, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Reconhecimento de Gestos - Pressiona q para sair", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
