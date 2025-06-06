import tensorflow as tf

# Caminho do modelo .h5 treinado
modelo_h5 = "modelo_gestos_lgp.h5"
# Nome de saída para o modelo convertido
modelo_tflite = "modelo_gestos_lgp.tflite"

# Carrega o modelo
modelo = tf.keras.models.load_model(modelo_h5)

# Converter para TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Otimização opcional
tflite_model = converter.convert()

# Salvar o modelo convertido
with open(modelo_tflite, "wb") as f:
    f.write(tflite_model)

print(f"✅ Modelo convertido e salvo como {modelo_tflite}")
