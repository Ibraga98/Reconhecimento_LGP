import cv2
import numpy as np
import mediapipe as mp
import time
import json
from tensorflow.keras.models import load_model

# Parâmetros
IMG_SIZE = 100  # Tamanho das imagens usado no treino
CONFIANCA_MINIMA = 0.6  # confiança mínima para aceitar um gesto
TEMPO_COOLDOWN = 2.0  # segundos de pausa entre gestos para evitar repetições

# Carregar o modelo Keras (melhor precisão que TFLite)
modelo = load_model("modelo_gestos_lgp.h5")
print(f"Formato de entrada do modelo: {IMG_SIZE}x{IMG_SIZE}")

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
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

# Instruções para o usuário
print("🖐️ Reconhecimento de Gestos com Tradução para Frases (Versão Melhorada)")
print("📷 Posicione sua mão na frente da câmera para reconhecer gestos")
print("🧠 Usando modelo Keras completo para melhor precisão")
print("⏱️ Sistema com pausa de {:.1f} segundos entre gestos para evitar repetições".format(TEMPO_COOLDOWN))
print("   - Aguarde a mensagem 'PRONTO para novo gesto' antes de fazer o próximo gesto")
print("   - Uma barra de progresso mostra o tempo restante de pausa")
print("⌨️ Comandos:")
print("   - Pressione 'q' para sair")
print("   - Pressione 'c' para limpar a frase atual e o histórico")
print("🔄 Uma nova frase começa automaticamente após 4 segundos sem gestos")
print("📊 Apenas gestos com confiança acima de {:.0f}% são adicionados à frase".format(CONFIANCA_MINIMA * 100))
print("🔍 Resultados detalhados são mostrados no terminal para debugging")

# Variáveis de controlo
frase_atual = []
todas_frases = []
ultimo_gesto = ""
tempo_ultimo_gesto = time.time()
TEMPO_PAUSA = 4  # segundos sem gesto para iniciar nova frase
em_cooldown = False  # indica se está em período de pausa entre gestos
tempo_fim_cooldown = 0  # quando termina o período de cooldown

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Espelhar a imagem para uma experiência mais intuitiva (como olhar no espelho)
    frame = cv2.flip(frame, 1)

    # Obter dimensões do frame para uso posterior
    h, w, _ = frame.shape

    # Verificar se o período de cooldown terminou
    tempo_atual = time.time()
    if em_cooldown and tempo_atual >= tempo_fim_cooldown:
        em_cooldown = False
        print("✅ Pronto para um novo gesto!")

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    gesto_predito = None
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        # Usar as dimensões já obtidas anteriormente
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
            # Pré-processamento (igual ao inferencia_tempo_real.py)
            imagem = cv2.resize(recorte, (IMG_SIZE, IMG_SIZE))
            imagem = imagem.astype("float32") / 255.0
            imagem = np.expand_dims(imagem, axis=0)

            # Predição com o modelo Keras
            pred = modelo.predict(imagem, verbose=0)
            indice_predito = np.argmax(pred)
            confianca = float(pred[0][indice_predito])
            gesto_predito = gestos[indice_predito].capitalize()

            # Mostrar resultado no terminal para debugging
            print(f"🖐 Gesto previsto: {gesto_predito} ({confianca:.2f})")

            tempo_atual = time.time()

            # Verificar se a confiança é suficiente para aceitar o gesto
            if confianca >= CONFIANCA_MINIMA and not em_cooldown:
                if gesto_predito != ultimo_gesto or (tempo_atual - tempo_ultimo_gesto) > 2:
                    if tempo_atual - tempo_ultimo_gesto > TEMPO_PAUSA and frase_atual:
                        todas_frases.append(" ".join(frase_atual))
                        frase_atual = []

                    frase_atual.append(gesto_predito)
                    ultimo_gesto = gesto_predito
                    tempo_ultimo_gesto = tempo_atual

                    # Ativar período de cooldown para dar tempo ao usuário
                    em_cooldown = True
                    tempo_fim_cooldown = tempo_atual + TEMPO_COOLDOWN
                    print(f"🕒 Pausa de {TEMPO_COOLDOWN} segundos para o próximo gesto...")
            else:
                # Mostrar que o gesto foi detectado mas com baixa confiança
                cv2.putText(frame, "Confiança baixa", (50, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            if len(frase_atual) > 10:
                todas_frases.append(" ".join(frase_atual))
                frase_atual = []

            # Mostrar gesto e confiança
            cv2.putText(frame, f"Gesto: {gesto_predito} ({confianca*100:.1f}%)", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar mensagem se nenhuma mão for detectada
    if not result.multi_hand_landmarks:
        cv2.putText(frame, "Nenhuma mão detectada", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostrar a frase atual
    frase_txt = " ".join(frase_atual)
    cv2.putText(frame, "Frase atual: " + frase_txt, (50, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Mostrar a frase anterior
    if todas_frases:
        ultima_frase = todas_frases[-1]
        cv2.putText(frame, "Frase anterior: " + ultima_frase, (50, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

    # Mostrar instruções na tela
    cv2.putText(frame, "q: Sair | c: Limpar frases", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Confiança mínima: {CONFIANCA_MINIMA*100:.0f}%", (10, h-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Mostrar status do cooldown
    if em_cooldown:
        tempo_restante = max(0, tempo_fim_cooldown - tempo_atual)
        cv2.putText(frame, f"AGUARDE: {tempo_restante:.1f}s para o próximo gesto", (50, 350), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Desenhar barra de progresso do cooldown
        largura_total = 300
        altura = 20
        progresso = int(largura_total * (1 - tempo_restante / TEMPO_COOLDOWN))
        cv2.rectangle(frame, (50, 360), (50 + largura_total, 360 + altura), (0, 0, 255), 2)
        cv2.rectangle(frame, (50, 360), (50 + progresso, 360 + altura), (0, 0, 255), -1)
    else:
        cv2.putText(frame, "PRONTO para novo gesto", (50, 350), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Reconhecimento de Gestos com Tradução (Com Pausa Entre Gestos)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        frase_atual = []
        todas_frases = []

cap.release()
cv2.destroyAllWindows()
