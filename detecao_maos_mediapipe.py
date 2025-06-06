import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Carrega o modelo treinado
modelo = load_model("modelo_gestos_lgp.h5")
gestos = ["nao", "ola", "sim"]

# Inicializa o MediaPipe
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils

# Iniciar captura
cap = cv2.VideoCapture(0)
with mp_maos.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(imagem_rgb)

        if resultado.multi_hand_landmarks:
            for mao in resultado.multi_hand_landmarks:
                mp_desenho.draw_landmarks(frame, mao, mp_maos.HAND_CONNECTIONS)

                # Extrai as coordenadas da mão
                altura, largura, _ = frame.shape
                x_coords = [int(p.x * largura) for p in mao.landmark]
                y_coords = [int(p.y * altura) for p in mao.landmark]

                # Define a caixa de recorte
                x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, largura)
                y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, altura)

                recorte = frame[y_min:y_max, x_min:x_max]
                if recorte.size == 0:
                    continue

                recorte_redimensionado = cv2.resize(recorte, (100, 100))
                entrada = np.expand_dims(recorte_redimensionado, axis=0) / 255.0

                # Predição
                pred = modelo.predict(entrada)
                gesto = gestos[np.argmax(pred)]
                confianca = np.max(pred)

                cv2.putText(frame, f"{gesto} ({confianca:.2f})", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Detecção com MediaPipe", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
