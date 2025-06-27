import cv2
import os
import time
import mediapipe as mp
import re

# Nome do gesto a capturar
NOME_GESTO = "nao"  # ← muda aqui para "nao", "ola", etc.
NUM_IMAGENS = 100
IMG_SIZE = 100
PASTA_DESTINO = f"dataset_limpo/{NOME_GESTO}"

# Criar pasta se não existir
os.makedirs(PASTA_DESTINO, exist_ok=True)

# Verificar se já existem imagens na pasta e encontrar o último número
ultimo_numero = 0
padrao = re.compile(f"{NOME_GESTO}_(\\d+)\\.jpg")
for arquivo in os.listdir(PASTA_DESTINO):
    match = padrao.match(arquivo)
    if match:
        numero = int(match.group(1))
        ultimo_numero = max(ultimo_numero, numero)

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Ativar webcam
cap = cv2.VideoCapture(0)
print(f"📷 A capturar imagens para: {NOME_GESTO} (Pressiona 'Espaço' para capturar, 'q' para sair)")
contador = ultimo_numero  # Começar a contar a partir do último número encontrado
total_a_capturar = ultimo_numero + NUM_IMAGENS  # Capturar NUM_IMAGENS novas imagens

print(f"Encontradas {ultimo_numero} imagens existentes. Serão capturadas mais {NUM_IMAGENS} imagens.")

while contador < total_a_capturar:
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

            # Desenhar e mostrar
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Mostrar progresso atual e total a capturar
            progresso_atual = contador - ultimo_numero
            cv2.putText(frame, f"Capturado {progresso_atual}/{NUM_IMAGENS} (Total: {contador}/{total_a_capturar})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Captura de Gestos - Pressiona Espaço para capturar, q para sair", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Captura ao pressionar espaço
        if results.multi_hand_landmarks:
            nome_arquivo = os.path.join(PASTA_DESTINO, f"{NOME_GESTO}_{contador+1}.jpg")
            cv2.imwrite(nome_arquivo, mao)
            contador += 1
            print(f"📸 Imagem {contador} capturada e salva em {nome_arquivo}")
    elif key == ord('q'):  # Sair ao pressionar 'q'
        break

cap.release()
cv2.destroyAllWindows()
imagens_capturadas = contador - ultimo_numero
print(f"✅ Captura finalizada. {imagens_capturadas} novas imagens capturadas nesta sessão.")
print(f"📊 Total de {contador} imagens disponíveis em {PASTA_DESTINO}")

# Notificar o usuário sobre a necessidade de treinar o modelo
if imagens_capturadas > 0:
    print("\n⚠️ IMPORTANTE: Como você capturou novas imagens, é necessário treinar o modelo novamente.")
    print("🔄 Execute 'python treinar_modelo.py' para atualizar o modelo com as novas imagens.")
