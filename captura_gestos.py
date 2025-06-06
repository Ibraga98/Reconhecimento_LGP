import cv2
import os
import time
import mediapipe as mp

# Nome do gesto a capturar
NOME_GESTO = "sim"  # ← muda aqui para "nao", "ola", etc.
NUM_IMAGENS = 100
IMG_SIZE = 100
PASTA_DESTINO = f"dataset_limpo/{NOME_GESTO}"

# Criar pasta se não existir
os.makedirs(PASTA_DESTINO, exist_ok=True)

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Ativar webcam
cap = cv2.VideoCapture(0)
print(f"📷 A capturar imagens para: {NOME_GESTO} (Pressiona 'q' para sair)")
contador = 0

while contador < NUM_IMAGENS:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            xmin, xmax = int(min(xs) * w), int(max(xs) * w)
            ymin, ymax = int(min(ys) * h), int(max(ys) * h)

            xmin = max(xmin - 20, 0)
            ymin = max(ymin - 20, 0)
            xmax = min(xmax + 20, w)
            ymax = min(ymax + 20, h)

            mao = frame[ymin:ymax, xmin:xmax]
            if mao.size == 0:
                continue

            mao = cv2.resize(mao, (IMG_SIZE, IMG_SIZE))
            nome_arquivo = os.path.join(PASTA_DESTINO, f"{NOME_GESTO}_{contador+1}.jpg")
            cv2.imwrite(nome_arquivo, mao)
            contador += 1

            # Desenhar e mostrar
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"Capturado {contador}/{NUM_IMAGENS}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Captura de Gestos - Pressiona q para sair", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Captura finalizada. {contador} imagens salvas em {PASTA_DESTINO}")
