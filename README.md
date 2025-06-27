🤟 Sistema de Reconhecimento de Gestos em Língua Gestual Portuguesa (LGP)

Este repositório apresenta um sistema desenvolvido para o reconhecimento de gestos em **Língua Gestual Portuguesa (LGP)**, com o objetivo de promover a inclusão social de pessoas surdas, facilitando a comunicação com ouvintes através da tradução de gestos em tempo real para texto.

O projeto foi desenvolvido por **Ivanilson Braga**, **Zakhar Khomyakivskyy** e **Ektiandro Elizabeth**, no âmbito da unidade curricular de Laboratorio De Projeto.



📋 Descrição Geral

A aplicação permite a deteção de gestos capturados por webcam, utilizando visão computacional e redes neurais convolucionais (CNN), e converte esses gestos em palavras escritas. A longo prazo, o sistema pode ser integrado a dispositivos móveis, aumentando o seu impacto social e acessibilidade.



📌 Palavras Atualmente Reconhecidas

O sistema foi treinado para reconhecer os seguintes gestos:

- **Olá**
- **Sim**
- **Não**
- **Água**
- **Por favor**
- **Bom dia**
  
Mais palavras podem ser adicionadas com facilidade através da captura de novos gestos e re-treinamento do modelo.



🛠 Tecnologias Utilizadas

- **Python 3.10+** – Linguagem principal de desenvolvimento
- **TensorFlow / Keras** – Construção e treino da rede neural
- **MediaPipe** – Deteção de mãos em tempo real
- **OpenCV** – Captura e visualização da webcam
- **Flutter** – Interface móvel Android (em desenvolvimento)
- **TensorFlow Lite** – Versão leve do modelo para dispositivos móveis



📂 Estrutura do Projeto

```bash
📁 Reconhecimento_LGP/
│
├── captura_gestos.py            # Captura imagens da webcam (só quando a mão é detetada)
├── limpar_dataset_externo.py    # Elimina imagens sem mãos
├── treinar_modelo.py            # Treina o modelo (.h5) com base nas imagens captadas
├── converter_para_tflite.py     # Converte o modelo para formato .tflite (mobile)
├── testar_modelo.py             # Testa a precisão do modelo com webcam
├── inferencia_tempo_real.py     # Reconhece gestos únicos em tempo real
├── inferencia_com_frase.py      # Reconhece frases com múltiplos gestos e pausas
│
├── modelo_gestos_lgp.h5         # Modelo treinado (Keras)
├── modelo_gestos_lgp.tflite     # Modelo convertido para Android
├── classes.json                 # Lista de classes/gestos reconhecidos
├── README.md                    # Ficheiro atual
└── dataset_limpo/               # Dataset final utilizado no treino, com imagens limpas
```



 🚀 Como Usar

 1. Capturar imagens da webcam
Executa o script e digita o nome do gesto (ex: "sim"):
```bash
python captura_gestos.py
```

 2. Limpar imagens sem mãos (automático com MediaPipe)
```bash
python limpar_dataset_externo.py
```

3. Treinar o modelo
```bash
python treinar_modelo.py
```

 4. Converter para TensorFlow Lite (.tflite)
```bash
python converter_para_tflite.py
```

 5. Testar o modelo com webcam
```bash
python testar_modelo.py
```

 6. Reconhecimento de frases com pausa
```bash
python inferencia_com_frase.py
```

---
🤖 Aplicação Android (em desenvolvimento)

A versão móvel da aplicação está a ser desenvolvida em **Flutter**, e utilizará o ficheiro `modelo_gestos_lgp.tflite` juntamente com `classes.json` para realizar inferência local no smartphone, sem necessidade de ligação à internet.

O processamento será todo feito no dispositivo, usando a câmara para interpretar os gestos em tempo real com baixa latência.



## 🧠 Como o modelo funciona?

O modelo baseia-se numa **CNN (Convolutional Neural Network)**, treinada com imagens captadas via webcam, que passaram por um processo de limpeza e normalização. Utiliza `categorical_crossentropy` como função de perda e `adam` como otimizador.

A entrada do modelo são imagens de 100x100 pixels, normalizadas. A saída é uma predição entre as classes configuradas em `classes.json`.



👥 Contribuidores

- **Ivanilson Braga**  
- **Zakhar Khomyakivskyy**  
- **Ektiandro Elizabeth**


🧾 Licença

Projeto desenvolvido para fins académicos no âmbito da unidade curricular de Projeto.  
Qualquer reutilização parcial ou total deve referenciar os autores.


💡 Considerações Finais

Este projeto mostrou-se eficaz na deteção e tradução de gestos da LGP para texto, abrindo portas para futuras expansões com mais sinais, integração por voz, suporte para outras línguas gestuais e interfaces multimodais. A integração com Flutter e a utilização do TensorFlow Lite tornam este sistema promissor para aplicação prática no quotidiano.

