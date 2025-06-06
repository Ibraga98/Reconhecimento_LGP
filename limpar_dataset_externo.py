import cv2
import os
import mediapipe as mp

# Caminhos
input_dir = "dataset"            # Pasta onde estão as imagens originais (incluindo iPhones)
output_dir = "dataset_limpo"     # Pasta onde serão salvas as imagens recortadas
IMG_SIZE = 100

# Inicializar MediaPipe
mp_maos = mp.solutions.hands
hands = mp_maos.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# Criar diretório de saída
os.makedirs(output_dir, exist_ok=True)

# Processar cada pasta (ex: nao, sim, ola)
for classe in os.listdir(input_dir):
    classe_path = os.path.join(input_dir, classe)
    if not os.path.isdir(classe_path):
        continue

    output_classe_path = os.path.join(output_dir, classe)
    os.makedirs(output_classe_path, exist_ok=True)

    for img_nome in os.listdir(classe_path):
        if not img_nome.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(classe_path, img_nome)
        imagem = cv2.imread(img_path)
        if imagem is None:
            continue

        img_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        resultado = hands.process(img_rgb)

        if resultado.multi_hand_landmarks:
            h, w, _ = imagem.shape
            pontos = resultado.multi_hand_landmarks[0].landmark
            x_coords = [int(p.x * w) for p in pontos]
            y_coords = [int(p.y * h) for p in pontos]
            x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, w)
            y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, h)

            recorte = imagem[y_min:y_max, x_min:x_max]
            if recorte.size > 0:
                recorte_resized = cv2.resize(recorte, (IMG_SIZE, IMG_SIZE))
                out_path = os.path.join(output_classe_path, img_nome)
                cv2.imwrite(out_path, recorte_resized)

print("✅ Limpeza concluída! Imagens salvas em:", output_dir)
