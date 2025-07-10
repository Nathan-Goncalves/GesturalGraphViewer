import math
import queue
import socket
import time
import webbrowser
from threading import Thread

import cv2
import dash
import mediapipe as mp
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output

camnum = 0

def create_graph():
    """Criar um grafo NetworkX 3D"""
    # Criar grafo aleatório geométrico 3D
    g = nx.random_geometric_graph(20, 0.3, dim=3)

    # Garantir que as posições estejam definidas
    pos = nx.get_node_attributes(g, 'pos')
    for node, position in pos.items():
        g.nodes[node]['pos'] = np.array(position)

    return g


def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def count_fingers(landmarks):
    """Conta dedos levantados considerando flexão leve"""
    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [3, 6, 10, 14, 18]

    fingers_up = 0

    # Polegar (comparação horizontal)
    thumb_tip = landmarks[finger_tips[0]]
    thumb_pip = landmarks[finger_pips[0]]
    wrist = landmarks[0]
    if thumb_tip.x > wrist.x:  # assume mão direita. Inverter sinal se for mão esquerda
        fingers_up += 1

    # Outros dedos (comparação vertical)
    for i in range(1, 5):
        tip = landmarks[finger_tips[i]]
        pip = landmarks[finger_pips[i]]
        if tip.y < pip.y * 0.95:  # Pequena margem para evitar ruídos
            fingers_up += 1

    return fingers_up

def is_closed_fist(landmarks):
    """Verifica se todos os dedos estão próximos ao punho"""
    wrist = landmarks[0]
    distances = []
    for tip_index in [4, 8, 12, 16, 20]:
        tip = landmarks[tip_index]
        distances.append(euclidean_distance(tip, wrist))

    avg_dist = sum(distances) / len(distances)
    return avg_dist < 0.1  # Ajuste sensível: entre 0.08 e 0.12

def detect_gestures(landmarks, debug=False):
    """
    Detecta gestos:
    - 'closed_fist' para mão fechada
    - 'open_hand' para mão aberta
    - 'none' se não identificar
    """
    if not landmarks or len(landmarks) < 21:
        return "none"

    try:
        if is_closed_fist(landmarks):
            if debug:
                print("Avg dist dedos-punho: CLOSED")
            return "closed_fist"

        fingers_up = count_fingers(landmarks)

        if debug:
            print(f"Dedos levantados: {fingers_up}")

        if fingers_up == 0:
            return "closed_fist"
        elif fingers_up >= 4:
            return "open_hand"
        else:
            return "none"
    except Exception as e:
        if debug:
            print("Erro na detecção:", e)
        return "none"
    
def find_free_port():
    """Encontrar uma porta livre"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port




class HandMotionGraph3D:

    @staticmethod
    def normalize_vector(v):
        #return v
        norm = math.sqrt(v['x']**2 + v['y']**2 + v['z']**2)
        if norm == 0:
            return v
        return {k: v[k]/norm for k in v}

    def __init__(self):
        # Inicializar MediaPipe
        self.camera_theta = None
        self.camera_radius = None
        self.camera_phi = None
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Configurações de vídeo
        self.cap = cv2.VideoCapture(camnum)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Criar grafo NetworkX
        self.graph = create_graph()

        # Controle da câmera/visualização 3D
        #self.camera_eye = {'x': 1.5, 'y': 1.5, 'z': 1.5}
        self.camera_eye = {'x': 1, 'y': 1, 'z': 1}


        self.camera_eye = self.normalize_vector(self.camera_eye) ###



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
                        'marginBottom': '20px',
                        'boxShadow': '0 4px 16px rgba(44,62,80,0.08)',
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
                                        'fontSize': '16px',
                                        'marginTop': '10px',
                                    })
                    ])
                ], style={
                    'width': '320px',
                    'minWidth': '240px',
                    'maxWidth': '350px',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'padding': '32px 24px 32px 24px',
                    'margin': '32px 24px 32px 0',
                    'backgroundColor': '#fff',
                    'borderRadius': '14px',
                    'boxShadow': '0 4px 24px rgba(44,62,80,0.10)',
                }),

                html.Div([
                    dcc.Graph(id='3d-graph', style={'height': '600px', 'backgroundColor': '#fafbfc', 'borderRadius': '8px', 'boxShadow': '0 2px 12px rgba(44,62,80,0.08)'}),
                ], style={
                    'width': 'calc(100% - 350px)',
                    'minWidth': '350px',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'padding': '32px 24px 32px 0',
                })
            ], style={
                'display': 'flex',
                'flexDirection': 'row',
                'justifyContent': 'center',
                'alignItems': 'flex-start',
                'width': '100%',
                'maxWidth': '1400px',
                'margin': '0 auto',
                'backgroundColor': '#f6f8fa',
                'borderRadius': '18px',
                'boxShadow': '0 2px 24px rgba(44,62,80,0.06)',
            }),

            # Componente oculto para atualização automática
            dcc.Interval(
                id='interval-component',
                interval=200,  # Atualizar a cada 200ms (5 FPS)
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
        """Obter informações de estatus"""
        status_color = '#27ae60' if self.hand_detected else '#e74c3c'
        gesture_icons = {
            'open_hand': '✋',
            'closed_fist': '✊',
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
                html.P("• Cima/Baixo: Zoom in/ Zoom out",
                       style={'margin': '2px 0 2px 20px', 'fontSize': '11px'}),
            ]),

            html.Div([
                html.P("✊ MÃO FECHADA - Movimento da Visualização:",
                       style={'margin': '5px 0', 'fontWeight': 'bold', 'color': '#3498db'}),
                html.P("• Move a visualização para onde a mão aponta",
                       style={'margin': '2px 0 2px 20px', 'fontSize': '11px'}),
            ]),

            html.Hr(),
            html.P(f"🔍 Zoom: {self.zoom_factor:.2f}", style={'margin': '2px 0'}),
            html.P(f"📊 Nós: {len(self.graph.nodes())}", style={'margin': '2px 0'}),
            html.P(f"📊 Arestas: {len(self.graph.edges())}", style={'margin': '2px 0'})
        ])

    def update_visualization_controls(self, hand_data, gesture):
        """Atualizar controles da visualização baseado na mão e gesto (modo extremidade: giro/pan/zoom)"""
        if not hand_data:
            return

        # Inicializar atributos esféricos se não existirem ou estiverem None
        if (not hasattr(self, 'camera_radius') or self.camera_radius is None or
            not hasattr(self, 'camera_theta') or self.camera_theta is None or
            not hasattr(self, 'camera_phi') or self.camera_phi is None):
            x, y, z = self.camera_eye['x'], self.camera_eye['y'], self.camera_eye['z']
            self.camera_radius = math.sqrt(x**2 + y**2 + z**2)
            self.camera_theta = math.atan2(y, math.sqrt(x**2 + z**2))
            self.camera_phi = math.atan2(x, z)

        x = hand_data['x']
        y = hand_data['y']
        pan_step = 0.12
        rot_step = 0.10
        zoom_step = 0.12

        # --- GIRO/ZOOM (mão aberta nas extremidades) ---
        if gesture == "open_hand":
            # Esquerda/Direita: gira azimute
            if x < 0.15:
                self.camera_phi -= rot_step
            elif x > 0.85:
                self.camera_phi += rot_step
            # Cima/Baixo: gira elevação
            if 0.25 <= y <= 0.75:
                pass  # Só gira se não estiver nas bordas verticais
            if y < 0.25:
                self.camera_theta -= rot_step
                # Zoom in se mão aberta no topo
                self.camera_radius = max(1.0, self.camera_radius - zoom_step)
                print(f'[ZOOM DEBUG] camera_radius (in): {self.camera_radius}')
            elif y > 0.75:
                self.camera_theta += rot_step
                # Zoom out se mão aberta na base
                self.camera_radius = min(30.0, self.camera_radius + zoom_step)
                print(f'[ZOOM DEBUG] camera_radius (out): {self.camera_radius}')
            # Limitar theta
            self.camera_theta = max(-math.pi/2 + 0.05, min(math.pi/2 - 0.05, self.camera_theta))
                # --- PAN (mão fechada nas extremidades) ---
            '''
            elif gesture == "closed_fist":
                if self.previous_hand_position:
                    dx = x - self.previous_hand_position['x']
                    dy = y - self.previous_hand_position['y']
                    self.camera_center['x'] -= dx * self.movement_sensitivity
                    self.camera_center['y'] += dy * self.movement_sensitivity
                    print(f"[PAN DEBUG] Δx: {dx:.3f}, Δy: {dy:.3f}")
            '''
        elif gesture == "closed_fist":
            moved = False
            if x < 0.25:
                self.camera_center['x'] -= pan_step
                moved = True
            elif x > 0.75:
                self.camera_center['x'] += pan_step
                moved = True
            if y < 0.25:
                self.camera_center['y'] += pan_step
                moved = True
            elif y > 0.75:
                self.camera_center['y'] -= pan_step
                moved = True
            if moved:
                print(f"[PAN DEBUG] camera_center: {self.camera_center}")
        ''''''

        # Atualizar posição da câmera (esférico → cartesiano)
        r = self.camera_radius * self.zoom_factor
        theta = self.camera_theta
        phi = self.camera_phi
        self.camera_eye['x'] = r * math.sin(theta) * math.sin(phi)
        self.camera_eye['y'] = r * math.cos(theta)
        self.camera_eye['z'] = r * math.sin(theta) * math.cos(phi)

        ''' '''
        new_eye_x = r * math.sin(theta) * math.sin(phi)
        new_eye_y = r * math.cos(theta)
        new_eye_z = r * math.sin(theta) * math.cos(phi)

        #### 
        '''
        if self.previous_hand_position:
            alpha = self.smoothing_factor
            self.camera_eye['x'] = (1 - alpha) * self.camera_eye['x'] + alpha * new_eye_x
            self.camera_eye['y'] = (1 - alpha) * self.camera_eye['y'] + alpha * new_eye_y
            self.camera_eye['z'] = (1 - alpha) * self.camera_eye['z'] + alpha * new_eye_z
        else:
            self.camera_eye['x'] = new_eye_x
            self.camera_eye['y'] = new_eye_y
            self.camera_eye['z'] = new_eye_z
        ####
        '''
        # Atualizar posição anterior (não usado para lógica de extremidade)
        self.previous_hand_position = {'x': x, 'y': y}

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
                gesture = detect_gestures(hand_landmarks.landmark)
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
        print("[RESET DEBUG] reset_visualization foi chamado!")
        #self.camera_eye = {'x': 1.5, 'y': 1.5, 'z': 1.5}


        self.camera_eye = {'x': 1, 'y': 1, 'z': 1}

        self.camera_eye = self.normalize_vector(self.camera_eye) ###



        self.camera_center = {'x': 0, 'y': 0, 'z': 0}
        self.camera_radius = 5.0
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
                    self.cleanup()
                    import sys
                    sys.exit(0)
                elif key == ord('r'):
                    self.reset_visualization()

            #time.sleep(0.033)  # ~30 FPS
            cv2.waitKey(1)

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
            port = find_free_port()

            print(f"🌐 Iniciando servidor web na porta {port}...")
            print(f"🔗 Acesse: http://localhost:{port}")

            # Abrir navegador automaticamente
            webbrowser.open(f'http://localhost:{port}')

            # Suprimir logs HTTP do Flask/Dash
            import logging
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)

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
        
        camnum = int(input("Digite o número da câmera (e.g.: 0): "))

        # Verificar se a câmera está disponível
        test_cap = cv2.VideoCapture(camnum)
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
