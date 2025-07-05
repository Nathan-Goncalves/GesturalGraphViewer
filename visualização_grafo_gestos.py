import cv2
import mediapipe as mp
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import threading
import time
from collections import deque
import math
import dash
from dash import dcc, html, Input, Output, callback
import json
import queue
import webbrowser
from threading import Thread
import socket


class HandMotionGraph3D:
    def __init__(self):
        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Configurações de vídeo
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Criar grafo NetworkX
        self.graph = self.create_graph()

        # Controle da câmera/visualização 3D
        self.camera_eye = {'x': 1.5, 'y': 1.5, 'z': 1.5}
        self.camera_center = {'x': 0, 'y': 0, 'z': 0}
        self.zoom_factor = 1.0

        # Suavização e histórico
        self.smoothing_factor = 0.3
        self.previous_hand_position = None

        # Estado da aplicação
        self.running = False
        self.hand_detected = False
        self.current_gesture = "none"
        self.hand_position = {'x': 0.5, 'y': 0.5}

        # Configurações de sensibilidade
        self.rotation_sensitivity = 3.0
        self.movement_sensitivity = 2.0
        self.zoom_sensitivity = 0.15

        # Queue para comunicação entre threads
        self.data_queue = queue.Queue(maxsize=10)

        # Dash app
        self.app = dash.Dash(__name__)
        self.setup_dash_app()

    def create_graph(self):
        """Criar um grafo NetworkX 3D"""
        # Criar grafo aleatório geométrico 3D
        G = nx.random_geometric_graph(20, 0.3, dim=3)

        # Garantir que as posições estejam definidas
        pos = nx.get_node_attributes(G, 'pos')
        for node, position in pos.items():
            G.nodes[node]['pos'] = np.array(position)

        return G

    def setup_dash_app(self):
        """Configurar aplicação Dash para visualização em tempo real"""
        self.app.layout = html.Div([
            html.H1("🤚 Controle de Grafo 3D com Gestos da Mão",
                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),

            html.Div([
                html.Div([
                    html.H3("Controles", style={'color': '#34495e'}),
                    html.Div(id='status-info', style={
                        'padding': '15px',
                        'backgroundColor': '#ecf0f1',
                        'borderRadius': '10px',
                        'marginBottom': '20px'
                    }),
                    html.Div([
                        html.Button('Reset Visualização', id='reset-btn',
                                    style={
                                        'backgroundColor': '#e74c3c',
                                        'color': 'white',
                                        'border': 'none',
                                        'padding': '10px 20px',
                                        'borderRadius': '5px',
                                        'cursor': 'pointer',
                                        'fontSize': '16px'
                                    })
                    ])
                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),

                html.Div([
                    dcc.Graph(id='3d-graph', style={'height': '600px'})
                ], style={'width': '75%', 'display': 'inline-block'})
            ]),

            # Componente oculto para atualização automática
            dcc.Interval(
                id='interval-component',
                interval=50,  # Atualizar a cada 50ms (20 FPS)
                n_intervals=0
            ),

            # Store para dados
            dcc.Store(id='reset-trigger', data=0)
        ])

        # Callbacks
        @self.app.callback(
            [Output('3d-graph', 'figure'),
             Output('status-info', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('reset-trigger', 'data')]
        )
        def update_plot_and_status(n, reset_count):
            return self.get_current_figure(), self.get_status_info()

        @self.app.callback(
            Output('reset-trigger', 'data'),
            [Input('reset-btn', 'n_clicks')],
            prevent_initial_call=True
        )
        def reset_position(n_clicks):
            if n_clicks:
                self.reset_visualization()
                return n_clicks
            return 0

    def get_current_figure(self):
        """Obter figura atual do Plotly com o grafo"""
        # Obter posições do grafo
        positions = nx.get_node_attributes(self.graph, 'pos')

        # Extrair coordenadas dos nós
        node_x = [positions[node][0] for node in self.graph.nodes()]
        node_y = [positions[node][1] for node in self.graph.nodes()]
        node_z = [positions[node][2] for node in self.graph.nodes()]

        # Extrair coordenadas das arestas
        edge_x, edge_y, edge_z = [], [], []
        for edge in self.graph.edges():
            node1, node2 = edge
            edge_x.extend([positions[node1][0], positions[node2][0], None])
            edge_y.extend([positions[node1][1], positions[node2][1], None])
            edge_z.extend([positions[node1][2], positions[node2][2], None])

        # Criar figura
        fig = go.Figure()

        # Adicionar arestas
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(width=3, color='gray'),
            name='Arestas',
            showlegend=False,
            hoverinfo='skip'
        ))

        # Cores dos nós baseadas no gesto
        node_colors = {
            'open_hand': '#2ecc71',  # Verde - Rotação
            'closed_fist': '#3498db',  # Azul - Movimento
            'pinch': '#f39c12',  # Laranja - Zoom
            'none': '#e74c3c'  # Vermelho - Nenhum
        }
        color = node_colors.get(self.current_gesture, '#e74c3c')

        # Adicionar nós
        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=8,
                color=color,
                opacity=0.8,
                line=dict(width=2, color='white')
            ),
            text=[f'Nó {i}' for i in self.graph.nodes()],
            name='Nós do Grafo',
            hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
        ))

        # Configurar layout com controle de câmera
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='Eixo X', gridcolor='lightgray'),
                yaxis=dict(title='Eixo Y', gridcolor='lightgray'),
                zaxis=dict(title='Eixo Z', gridcolor='lightgray'),
                bgcolor='rgba(240,240,240,0.1)',
                camera=dict(
                    eye=dict(
                        x=self.camera_eye['x'] * self.zoom_factor,
                        y=self.camera_eye['y'] * self.zoom_factor,
                        z=self.camera_eye['z'] * self.zoom_factor
                    ),
                    center=dict(
                        x=self.camera_center['x'],
                        y=self.camera_center['y'],
                        z=self.camera_center['z']
                    )
                ),
                aspectmode='cube'
            ),
            title=f'Grafo 3D - Gesto: {self.current_gesture} | Zoom: {self.zoom_factor:.2f}',
            showlegend=True,
            margin=dict(l=0, r=0, t=40, b=0),
            font=dict(size=12)
        )

        return fig

    def get_status_info(self):
        """Obter informações de status"""
        status_color = '#27ae60' if self.hand_detected else '#e74c3c'
        gesture_icons = {
            'open_hand': '✋',
            'closed_fist': '✊',
            'pinch': '🤏',
            'none': '❌'
        }

        # Determinar posição da mão
        hand_pos_text = ""
        if self.hand_detected:
            if self.hand_position['x'] < 0.3:
                hand_pos_text = "← Esquerda"
            elif self.hand_position['x'] > 0.7:
                hand_pos_text = "→ Direita"
            else:
                hand_pos_text = "↔ Centro"

            if self.hand_position['y'] < 0.3:
                hand_pos_text += " ↑ Cima"
            elif self.hand_position['y'] > 0.7:
                hand_pos_text += " ↓ Baixo"

        return html.Div([
            html.P(f"🎯 Mão Detectada: ", style={'margin': '5px 0', 'fontWeight': 'bold'}),
            html.Span(f"{'Sim' if self.hand_detected else 'Não'}",
                      style={'color': status_color, 'fontWeight': 'bold'}),

            html.P(f"👋 Gesto: {gesture_icons.get(self.current_gesture, '❓')} {self.current_gesture}",
                   style={'margin': '5px 0'}),

            html.P(f"📍 Posição da Mão: {hand_pos_text}",
                   style={'margin': '5px 0', 'fontSize': '12px'}),

            html.Hr(),
            html.H4("Controles:", style={'margin': '10px 0', 'color': '#34495e', 'fontSize': '14px'}),

            html.Div([
                html.P("✋ MÃO ABERTA - Rotação da Visualização:",
                       style={'margin': '5px 0', 'fontWeight': 'bold', 'color': '#2ecc71'}),
                html.P("• Esquerda/Direita: Rotaciona horizontalmente",
                       style={'margin': '2px 0 2px 20px', 'fontSize': '11px'}),
                html.P("• Cima/Baixo: Rotaciona verticalmente",
                       style={'margin': '2px 0 2px 20px', 'fontSize': '11px'}),
            ]),

            html.Div([
                html.P("✊ MÃO FECHADA - Movimento da Visualização:",
                       style={'margin': '5px 0', 'fontWeight': 'bold', 'color': '#3498db'}),
                html.P("• Move a visualização para onde a mão aponta",
                       style={'margin': '2px 0 2px 20px', 'fontSize': '11px'}),
            ]),

            html.Div([
                html.P("🤏 PINÇA - Zoom:",
                       style={'margin': '5px 0', 'fontWeight': 'bold', 'color': '#f39c12'}),
                html.P("• Fechar pinça: Zoom in",
                       style={'margin': '2px 0 2px 20px', 'fontSize': '11px'}),
                html.P("• Abrir pinça: Zoom out",
                       style={'margin': '2px 0 2px 20px', 'fontSize': '11px'}),
            ]),

            html.Hr(),
            html.P(f"🔍 Zoom: {self.zoom_factor:.2f}", style={'margin': '2px 0'}),
            html.P(f"📊 Nós: {len(self.graph.nodes())}", style={'margin': '2px 0'}),
            html.P(f"📊 Arestas: {len(self.graph.edges())}", style={'margin': '2px 0'})
        ])

    def detect_gestures(self, landmarks):
        """Detectar gestos específicos"""
        if not landmarks:
            return "none"

        try:
            # Contar dedos levantados
            fingers_up = self.count_fingers(landmarks)

            # Detectar pinça (polegar e indicador próximos)
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            pinch_distance = math.sqrt(
                (thumb_tip.x - index_tip.x) ** 2 +
                (thumb_tip.y - index_tip.y) ** 2
            )

            # Se pinça está sendo feita (dedos muito próximos)
            if pinch_distance < 0.05:
                return "pinch"
            elif fingers_up == 0:
                return "closed_fist"
            elif fingers_up >= 4:
                return "open_hand"
            else:
                return "none"

        except:
            return "none"

    def count_fingers(self, landmarks):
        """Contar dedos levantados"""
        finger_tips = [4, 8, 12, 16, 20]  # Pontas dos dedos
        finger_pips = [3, 6, 10, 14, 18]  # Articulações intermediárias

        fingers_up = 0

        # Polegar (comparação horizontal)
        if landmarks[finger_tips[0]].x > landmarks[finger_pips[0]].x:
            fingers_up += 1

        # Outros dedos (comparação vertical)
        for i in range(1, 5):
            if landmarks[finger_tips[i]].y < landmarks[finger_pips[i]].y:
                fingers_up += 1

        return fingers_up

    def get_pinch_distance(self, landmarks):
        """Calcular distância da pinça para controle de zoom"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]

        distance = math.sqrt(
            (thumb_tip.x - index_tip.x) ** 2 +
            (thumb_tip.y - index_tip.y) ** 2 +
            (thumb_tip.z - index_tip.z) ** 2
        )

        return distance

    def update_visualization_controls(self, hand_data, gesture):
        """Atualizar controles da visualização baseado na mão e gesto"""
        if not hand_data:
            return

        # Posição atual da mão
        current_x = hand_data['x']
        current_y = hand_data['y']

        # Calcular movimento se houver posição anterior
        if self.previous_hand_position:
            dx = current_x - self.previous_hand_position['x']
            dy = current_y - self.previous_hand_position['y']

            # Aplicar suavização
            dx *= self.smoothing_factor
            dy *= self.smoothing_factor

            if gesture == "open_hand":
                # MÃO ABERTA: Rotação da visualização
                # Movimento horizontal da mão = rotação horizontal (ao redor do eixo Y)
                # Movimento vertical da mão = rotação vertical (ao redor do eixo X)

                # Rotação horizontal (esquerda/direita)
                if abs(dx) > 0.001:
                    angle_y = dx * self.rotation_sensitivity
                    # Rotacionar câmera ao redor do eixo Y
                    cos_y = math.cos(angle_y)
                    sin_y = math.sin(angle_y)

                    new_x = self.camera_eye['x'] * cos_y - self.camera_eye['z'] * sin_y
                    new_z = self.camera_eye['x'] * sin_y + self.camera_eye['z'] * cos_y

                    self.camera_eye['x'] = new_x
                    self.camera_eye['z'] = new_z

                # Rotação vertical (cima/baixo)
                if abs(dy) > 0.001:
                    angle_x = -dy * self.rotation_sensitivity
                    # Rotacionar câmera ao redor do eixo X
                    cos_x = math.cos(angle_x)
                    sin_x = math.sin(angle_x)

                    new_y = self.camera_eye['y'] * cos_x - self.camera_eye['z'] * sin_x
                    new_z = self.camera_eye['y'] * sin_x + self.camera_eye['z'] * cos_x

                    self.camera_eye['y'] = new_y
                    self.camera_eye['z'] = new_z

            elif gesture == "closed_fist":
                # MÃO FECHADA: Movimento da visualização
                # Move o centro da visualização para onde a mão aponta
                self.camera_center['x'] += dx * self.movement_sensitivity
                self.camera_center['y'] -= dy * self.movement_sensitivity  # Inverter Y

        # Controle de zoom com pinça
        if gesture == "pinch" and 'pinch_distance' in hand_data:
            if hasattr(self, 'previous_pinch_distance'):
                # Calcular mudança na distância da pinça
                distance_change = hand_data['pinch_distance'] - self.previous_pinch_distance

                # Aplicar zoom baseado na mudança da distância
                if abs(distance_change) > 0.001:
                    zoom_change = distance_change * self.zoom_sensitivity
                    self.zoom_factor = max(0.1, min(5.0, self.zoom_factor + zoom_change))

            self.previous_pinch_distance = hand_data['pinch_distance']

        # Atualizar posição anterior
        self.previous_hand_position = {'x': current_x, 'y': current_y}

    def process_frame(self):
        """Processar frame da câmera"""
        ret, frame = self.cap.read()
        if not ret:
            return None, False

        # Espelhar horizontalmente
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processar com MediaPipe
        results = self.hands.process(rgb_frame)

        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True

            for hand_landmarks in results.multi_hand_landmarks:
                # Desenhar landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                # Detectar gesto
                gesture = self.detect_gestures(hand_landmarks.landmark)
                self.current_gesture = gesture

                # Extrair posição da mão (usar ponto do pulso como referência)
                wrist = hand_landmarks.landmark[0]
                hand_data = {
                    'x': wrist.x,
                    'y': wrist.y,
                    'z': wrist.z
                }

                # Atualizar posição da mão para exibição
                self.hand_position = {'x': wrist.x, 'y': wrist.y}

                # Se for pinça, adicionar distância
                if gesture == "pinch":
                    hand_data['pinch_distance'] = self.get_pinch_distance(hand_landmarks.landmark)

                # Atualizar controles da visualização
                self.update_visualization_controls(hand_data, gesture)

        else:
            self.current_gesture = "none"

        self.hand_detected = hand_detected

        # Adicionar informações no frame
        cv2.putText(frame, f"Mao detectada: {'Sim' if hand_detected else 'Nao'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if hand_detected else (0, 0, 255), 2)
        cv2.putText(frame, f"Gesto: {self.current_gesture}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Mostrar posição da mão
        if hand_detected:
            pos_text = ""
            if self.hand_position['x'] < 0.3:
                pos_text += "ESQ "
            elif self.hand_position['x'] > 0.7:
                pos_text += "DIR "
            if self.hand_position['y'] < 0.3:
                pos_text += "CIMA"
            elif self.hand_position['y'] > 0.7:
                pos_text += "BAIXO"
            cv2.putText(frame, f"Posicao: {pos_text}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, f"Zoom: {self.zoom_factor:.2f}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Instruções
        cv2.putText(frame, "Controles:",
                    (10, frame.shape[0] - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, "Mao aberta: Rotacao da visualizacao",
                    (10, frame.shape[0] - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, "Mao fechada: Movimento da visualizacao",
                    (10, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, "Pinça (polegar+indicador): Zoom",
                    (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, "Pressione 'q' para sair",
                    (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, "Pressione 'r' para resetar",
                    (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return frame, hand_detected

    def reset_visualization(self):
        """Resetar visualização do grafo"""
        self.camera_eye = {'x': 1.5, 'y': 1.5, 'z': 1.5}
        self.camera_center = {'x': 0, 'y': 0, 'z': 0}
        self.zoom_factor = 1.0
        self.current_gesture = "none"
        self.previous_hand_position = None
        if hasattr(self, 'previous_pinch_distance'):
            delattr(self, 'previous_pinch_distance')
        print("✅ Visualização resetada!")

    def camera_thread(self):
        """Thread para processar câmera"""
        print("📹 Iniciando thread da câmera...")

        while self.running:
            frame, hand_detected = self.process_frame()

            if frame is not None:
                cv2.imshow('Hand Motion Graph Control - Camera Feed', frame)

                # Verificar teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('r'):
                    self.reset_visualization()

            time.sleep(0.033)  # ~30 FPS

    def find_free_port(self):
        """Encontrar uma porta livre"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def run(self):
        """Executar aplicação principal"""
        print("🤚 Iniciando controle 3D de grafo com gestos da mão...")
        print("=" * 70)
        print("📋 NOVOS CONTROLES:")
        print("1. ✋ MÃO ABERTA - Rotação da Visualização:")
        print("   • Mover mão para ESQUERDA/DIREITA = Rotaciona horizontalmente")
        print("   • Mover mão para CIMA/BAIXO = Rotaciona verticalmente")
        print()
        print("2. ✊ MÃO FECHADA - Movimento da Visualização:")
        print("   • Move a visualização para onde a mão estiver apontando")
        print("   • ESQUERDA/DIREITA/CIMA/BAIXO = Move visualização")
        print()
        print("3. 🤏 PINÇA (polegar + indicador) - Zoom:")
        print("   • Fechar a pinça = Zoom IN (aproximar)")
        print("   • Abrir a pinça = Zoom OUT (afastar)")
        print()
        print("4. Teclas:")
        print("   • 'q' = Sair")
        print("   • 'r' = Resetar visualização")
        print("=" * 70)

        # Verificar se a câmera está disponível
        if not self.cap.isOpened():
            print("❌ Erro: Não foi possível acessar a câmera")
            return

        self.running = True

        try:
            # Iniciar thread da câmera
            camera_thread = Thread(target=self.camera_thread, daemon=True)
            camera_thread.start()

            # Encontrar porta livre
            port = self.find_free_port()

            print(f"🌐 Iniciando servidor web na porta {port}...")
            print(f"🔗 Acesse: http://localhost:{port}")

            # Abrir navegador automaticamente
            webbrowser.open(f'http://localhost:{port}')

            # Executar Dash app
            self.app.run(debug=False, port=port, host='0.0.0.0')

        except KeyboardInterrupt:
            print("\n⚠️ Interrompido pelo usuário")
        except Exception as e:
            print(f"❌ Erro: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Limpar recursos"""
        print("🧹 Limpando recursos...")
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Recursos liberados. Até logo! 👋")


def main():
    """Função principal"""
    try:
        # Verificar se a câmera está disponível
        test_cap = cv2.VideoCapture(0)
        if not test_cap.isOpened():
            print("❌ Erro: Não foi possível acessar a câmera")
            print("💡 Certifique-se de que:")
            print("   - A câmera está conectada")
            print("   - Nenhum outro aplicativo está usando a câmera")
            print("   - Você tem permissão para acessar a câmera")
            return
        test_cap.release()

        # Criar e executar aplicação
        app = HandMotionGraph3D()
        app.run()

    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        print("\n💡 Para instalar as dependências necessárias, execute:")
        print("pip install opencv-python mediapipe plotly numpy dash networkx")
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")


if __name__ == "__main__":
    main()