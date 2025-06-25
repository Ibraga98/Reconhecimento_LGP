import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import json

# Carregar o modelo TFLite
interpreter = tf.lite.Interpreter(model_path="modelo_gestos_lgp.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]  # (altura, largura)

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Carregar gestos do arquivo classes.json
with open("classes.json", "r") as f:
    class_indices = json.load(f)

# Criar lista ordenada de gestos
gestos = [None] * len(class_indices)
for classe, idx in class_indices.items():
    gestos[idx] = classe

# Webcam
cap = cv2.VideoCapture(0)

# Variáveis de controlo
frase_atual = []
todas_frases = []
ultimo_gesto = ""
tempo_ultimo_gesto = time.time()
TEMPO_PAUSA = 4  # segundos sem gesto para iniciar nova frase

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    gesto_predito = None
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        h, w, _ = frame.shape
        x_coords = [lm.x * w for lm in result.multi_hand_landmarks[0].landmark]
        y_coords = [lm.y * h for lm in result.multi_hand_landmarks[0].landmark]
        xmin, xmax = int(min(x_coords)), int(max(x_coords))
        ymin, ymax = int(min(y_coords)), int(max(y_coords))
        margem = 20

        xmin = max(0, xmin - margem)
        ymin = max(0, ymin - margem)
        xmax = min(w, xmax + margem)
        ymax = min(h, ymax + margem)

        recorte = frame[ymin:ymax, xmin:xmax]
        if recorte.size > 0:
            imagem = cv2.resize(recorte, tuple(input_shape))
            imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
            imagem = imagem.astype(np.float32) / 255.0
            imagem = np.expand_dims(imagem, axis=0)

            interpreter.set_tensor(input_details[0]['index'], imagem)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            indice_predito = np.argmax(output_data)
            gesto_predito = gestos[indice_predito].capitalize()

            tempo_atual = time.time()

            if gesto_predito != ultimo_gesto or (tempo_atual - tempo_ultimo_gesto) > 2:
                if tempo_atual - tempo_ultimo_gesto > TEMPO_PAUSA and frase_atual:
                    todas_frases.append(" ".join(frase_atual))
                    frase_atual = []

                frase_atual.append(gesto_predito)
                ultimo_gesto = gesto_predito
                tempo_ultimo_gesto = tempo_atual

            if len(frase_atual) > 10:
                todas_frases.append(" ".join(frase_atual))
                frase_atual = []

            cv2.putText(frame, f"Gesto: {gesto_predito}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar a frase atual
    frase_txt = " ".join(frase_atual)
    cv2.putText(frame, "Frase atual: " + frase_txt, (50, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Mostrar a frase anterior
    if todas_frases:
        ultima_frase = todas_frases[-1]
        cv2.putText(frame, "Frase anterior: " + ultima_frase, (50, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

    cv2.imshow("Reconhecimento de Gestos com Tradução", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        frase_atual = []
        todas_frases = []

cap.release()
cv2.destroyAllWindows()
