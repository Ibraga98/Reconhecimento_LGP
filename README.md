# 🤖 Projeto de Reconhecimento de Gestos LGP (Língua Gestual Portuguesa)

Este projeto tem como objetivo reconhecer gestos da LGP através de uma webcam, utilizando um modelo de visão computacional treinado com TensorFlow/Keras e posteriormente convertido para utilização em uma aplicação Android com Flutter.

---

## 📦 Estrutura do Projeto

```
├── captura_gestos.py                # Captura imagens para treino
├── treinar_modelo.py               # Treina modelo CNN com imagens do dataset
├── modelo_gestos_lgp.h5            # Modelo treinado em Keras
├── modelo_gestos_lgp.tflite        # Modelo convertido para uso em Android
├── classes.json                    # Mapeamento de classes (gestos)
├── inferencia_tempo_real.py       # Testes em tempo real com webcam
├── detecao_maos_mediapipe.py      # Módulo de deteção de mãos com MediaPipe
├── testar_modelo.py               # Verifica a performance do modelo treinado
├── dataset_limpo/                 # Dataset organizado por classes
├── modelo_android/                # Pasta específica para integração com Android
│   ├── modelo_gestos_lgp.tflite
│   ├── classes.json
│   ├── instrucoes.txt
│   └── exemplo_flutter_tflite.dart
```

---

## 🧠 Modelo

- Modelo CNN simples com `Conv2D`, `MaxPooling2D`, `Flatten` e `Dense`
- Treinado com imagens 100x100px RGB normalizadas
- Exportado para `.tflite` com `TFLiteConverter`

---

## 🖥️ Execução no PC

```bash
python captura_gestos.py           # Captura imagens
python treinar_modelo.py          # Treina o modelo
python converter_para_tflite.py   # Converte o modelo para .tflite
python inferencia_tempo_real.py   # Testa em tempo real
```

---

## 📱 Integração Android

A pasta `modelo_android/` contém:

- `modelo_gestos_lgp.tflite`
- `classes.json`
- `instrucoes.txt`
- `exemplo_flutter_tflite.dart`

Utilizar `tflite_flutter` no Flutter para carregar e usar o modelo.

---

## 👨‍💻 Autores

- 👤 Ivanilson Braga — Implementação em Python, treino do modelo, deteção de mãos
- 👤 Zakhar Khomyakivskyy Integração com Flutter e desenvolvimento Android

---

## ✅ Estado Atual

✔️ Dataset personalizado criado  
✔️ Modelo treinado com alta precisão  
✔️ Deteção de mãos em tempo real integrada  
✔️ Código limpo e documentado  
⏳ Pronto para integração no Android
