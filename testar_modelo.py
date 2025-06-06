import cv2
import numpy as np
import json
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

# Ativar webcam
cap = cv2.VideoCapture(0)
print("📷 Webcam ativa. Pressiona 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Imagem espelhada para visualização
    frame_visivel = cv2.flip(frame, 1)

    # Pré-processamento da imagem para o modelo
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
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
    cv2.putText(frame_visivel, texto, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow('Reconhecimento de Gestos - Pressiona q para sair', frame_visivel)

    # Fechar com tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
