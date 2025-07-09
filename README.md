# GesturalGraphViewer

Visualize e controle um grafo 3D em tempo real usando gestos das mãos capturados pela webcam!

## Recursos principais
- Visualização 3D interativa de grafos com Plotly e Dash
- Controle total da câmera (zoom, rotação, pan) usando gestos naturais
- Detecção de mão em tempo real via MediaPipe
- Interface web moderna e responsiva
- Feedback visual dos gestos reconhecidos

## Como usar

### 1. Instale as dependências

Recomenda-se o uso de um ambiente virtual (venv ou conda):

```bash
pip install -r requirements.txt
```

### 2. Execute o aplicativo

```bash
python visualização_grafo_gestos.py
```

Acesse o endereço exibido no terminal (ex: http://localhost:8050) para abrir a interface web.

### 3. Controles por gesto

- **✋ Mão aberta:**
  - **Topo da tela:** Zoom in (aproxima o grafo)
  - **Base da tela:** Zoom out (afasta o grafo)
  - **Lateral esquerda/direita:** Rotaciona o grafo
- **✊ Mão fechada:**
  - **Extremidades:** Pan/move a visualização do grafo

> **Dica:** O feedback do gesto atual aparece na interface!

### 4. Encerrando
- Pressione a tecla `q` na janela da webcam para encerrar completamente o sistema.

## Requisitos
- Python 3.8+
- Webcam funcional
- Navegador moderno (Chrome, Firefox, Edge, etc)

## Principais dependências
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://google.github.io/mediapipe/)
- [Plotly](https://plotly.com/python/)
- [Dash](https://dash.plotly.com/)
- [NetworkX](https://networkx.org/)
- [NumPy](https://numpy.org/)

## Estrutura dos arquivos principais
- `visualização_grafo_gestos.py`: código principal, lógica dos gestos e interface web
- `requirements.txt`: dependências do projeto

## Personalização
- Para alterar a sensibilidade dos gestos, ajuste os parâmetros `zoom_step`, `pan_step` e os limites de detecção no código.
- Para mudar o grafo, edite a função `create_graph()`.



---

Sinta-se à vontade para relatar bugs, sugerir melhorias ou adaptar para outros tipos de visualização!
