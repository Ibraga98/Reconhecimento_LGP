import os
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Caminho para o dataset limpo
dataset_path = "dataset_limpo"

# Parâmetros
IMG_SIZE = 100
BATCH_SIZE = 16
EPOCHS = 30
VALIDATION_SPLIT = 0.2

# Geradores de imagem com divisão treino/validação
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=VALIDATION_SPLIT
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Imprimir informações sobre os dados carregados
# Isso confirma que todas as imagens estão sendo utilizadas, mesmo que sejam mais de 100 por classe
print(f"Número de amostras de treino: {train_generator.samples}")
print(f"Número de amostras de validação: {val_generator.samples}")
print(f"Número total de amostras: {train_generator.samples + val_generator.samples}")
print(f"Número de classes: {train_generator.num_classes}")
print(f"Nomes das classes: {train_generator.class_indices}")

# Verificar número de imagens por classe
# Isso mostra o número total de imagens disponíveis em cada diretório de classe
import os
for classe in train_generator.class_indices.keys():
    caminho_classe = os.path.join(dataset_path, classe)
    num_imagens = len([f for f in os.listdir(caminho_classe) if os.path.isfile(os.path.join(caminho_classe, f))])
    print(f"Classe '{classe}': {num_imagens} imagens no diretório")

# Guardar as classes num ficheiro JSON
with open("classes.json", "w") as f:
    json.dump(train_generator.class_indices, f)

# Modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compilar modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("modelo_gestos_lgp.h5", save_best_only=True)

# Treinar
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

print("✅ Treino concluído. Modelo salvo como modelo_gestos_lgp.h5")
print("📊 Todas as imagens do dataset foram utilizadas no treino, incluindo aquelas além das 100 primeiras por classe.")
