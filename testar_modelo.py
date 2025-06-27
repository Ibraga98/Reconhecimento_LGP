import cv2
import numpy as np
import json
import mediapipe as mp
from tensorflow.keras.models import load_model

# Parâmetros
IMG_SIZE = 100

# Carregar nomes das classes automaticamente a partir do ficheiro gerado no treino
with open("classes.json", "r") as f:
    class_indices = json.load(f)

# Reorganizar para lista ordenada pela posição correta
GESTOS = [None] * len(class_indices)
for classe, idx in class_indices.items():
    GESTOS[idx] = classe

# Carregar o modelo treinado
modelo = load_model('modelo_gestos_lgp.h5')

# Inicializar MediaPipe para deteção de mãos
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils

# Ativar webcam
cap = cv2.VideoCapture(0)
print("📷 Webcam ativa. Pressiona 'q' para sair.")

# Iniciar processamento com MediaPipe
with mp_maos.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Imagem espelhada para visualização
        frame = cv2.flip(frame, 1)

        # Converter para RGB (necessário para o MediaPipe)
        imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processar a imagem com MediaPipe
        resultado = hands.process(imagem_rgb)

        # Variáveis para armazenar predição
        gesto_previsto = "Nenhuma mão detetada"
        confianca = 0.0

        # Se mãos forem detetadas
        if resultado.multi_hand_landmarks:
            for mao in resultado.multi_hand_landmarks:
                # Desenhar os pontos da mão
                mp_desenho.draw_landmarks(frame, mao, mp_maos.HAND_CONNECTIONS)

                # Extrair coordenadas da mão
                altura, largura, _ = frame.shape
                x_coords = [int(p.x * largura) for p in mao.landmark]
                y_coords = [int(p.y * altura) for p in mao.landmark]

                # Definir a caixa de recorte com margem
                x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, largura)
                y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, altura)

                # Recortar a região da mão
                recorte = frame[y_min:y_max, x_min:x_max]
                if recorte.size == 0:
                    continue

                # Pré-processamento da imagem para o modelo
                recorte_redimensionado = cv2.resize(recorte, (IMG_SIZE, IMG_SIZE))
                img = recorte_redimensionado.astype("float32") / 255.0
                img = np.expand_dims(img, axis=0)

                # Fazer predição
                pred = modelo.predict(img, verbose=0)
                gesto_idx = np.argmax(pred)
                confianca = np.max(pred)
                gesto_previsto = GESTOS[gesto_idx]

                # Mostrar resultado no terminal
                print(f"🖐 Gesto previsto: {gesto_previsto} ({confianca:.2f})")

                # Mostrar na janela com texto
                texto = f"{gesto_previsto.upper()} ({confianca*100:.1f}%)"
                cv2.putText(frame, texto, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Mostrar mensagem se nenhuma mão for detetada
        else:
            cv2.putText(frame, "Nenhuma mão detetada", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow('Reconhecimento de Gestos com MediaPipe - Pressiona q para sair', frame)

        # Fechar com tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
