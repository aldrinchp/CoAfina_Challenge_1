import streamlit as st
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import deque
import torch
from transformers import AutoProcessor, SiglipVisionModel
from sklearn.cluster import KMeans
import umap.umap_ as umap
import tempfile
import os
from deep_sort_realtime.deepsort_tracker import DeepSort

# Configuración de página
st.set_page_config(
    page_title="Bytelab AI",
    page_icon="⚽",
    layout="wide"
)

# --- CACHÉ DE MODELOS ---
@st.cache_resource
def load_yolo_models():
    """Carga modelos YOLO una sola vez"""
    with st.spinner("🔄 Cargando modelos YOLO..."):
        model_vertex = YOLO("Models/vertex_little.pt")
        model_players = YOLO("Models/pgrb_little.pt")
        model_ball = YOLO("Models/ball_little.pt")
    return model_vertex, model_players, model_ball
from transformers import CLIPProcessor, CLIPModel

@st.cache_resource
def load_siglip_model():
    """Carga un modelo CLIP más liviano compatible con Streamlit Cloud"""
    with st.spinner("🔄 Cargando modelo CLIP..."):
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        CLIP_MODEL_PATH = "openai/clip-vit-base-patch32"
        embeddings_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(DEVICE)
        embeddings_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
    return embeddings_model, embeddings_processor, DEVICE


# --- CONSTANTES ---
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

TEAM_COLORS = {
    0: (0, 0, 255),              # Rojo - Equipo 0
    1: (255, 0, 0),              # Azul - Equipo 1
    'goalkeeper': (0, 255, 255),  # Amarillo - Porteros
    'referee': (128, 128, 128)    # Gris - Árbitros
}

# --- CLASES ---
class BallTracker:
    def __init__(self, buffer_size: int = 20):
        self.buffer = deque(maxlen=buffer_size)
        self.last_position = None

    def update(self, detections: sv.Detections) -> tuple:
        if len(detections) == 0:
            return None

        xy = detections.get_anchors_coordinates(sv.Position.CENTER)

        if self.last_position is None:
            self.last_position = xy[0]
        else:
            distances = np.linalg.norm(xy - self.last_position, axis=1)
            index = np.argmin(distances)
            self.last_position = xy[index]

        self.buffer.append(self.last_position)
        return self.last_position


class TeamClassifier:
    def __init__(self, embeddings_model, embeddings_processor, device):
        self.embeddings_model = embeddings_model
        self.embeddings_processor = embeddings_processor
        self.device = device
        self.is_fitted = False
        self.reducer = None
        self.kmeans = None
        
    def extract_embeddings(self, crops, batch_size=16):
        """Extrae embeddings usando SigLIP"""
        if len(crops) == 0:
            return np.array([])
        
        crops_pil = [sv.cv2_to_pillow(crop) for crop in crops]
        
        embeddings_list = []
        with torch.no_grad():
            for i in range(0, len(crops_pil), batch_size):
                batch = crops_pil[i:i + batch_size]
                inputs = self.embeddings_processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.embeddings_model(**inputs)
                batch_embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                embeddings_list.append(batch_embeddings)
        
        return np.concatenate(embeddings_list, axis=0)
        
    def fit(self, crops, progress_callback=None):
        """Entrena el clasificador con crops de jugadores"""
        if len(crops) < 20:
            return False, f"Insuficientes crops ({len(crops)}). Se necesitan al menos 20."
        
        if progress_callback:
            progress_callback(0.1, "Extrayendo embeddings...")
        
        embeddings = self.extract_embeddings(crops, batch_size=16)
        
        if progress_callback:
            progress_callback(0.5, "Reduciendo dimensionalidad (UMAP)...")
        
        self.reducer = umap.UMAP(n_components=3, random_state=42)
        projections = self.reducer.fit_transform(embeddings)
        
        if progress_callback:
            progress_callback(0.8, "Aplicando K-Means clustering...")
        
        self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
        clusters = self.kmeans.fit_predict(projections)
        
        self.is_fitted = True
        
        team0_count = (clusters == 0).sum()
        team1_count = (clusters == 1).sum()
        
        if progress_callback:
            progress_callback(1.0, "¡Clasificador entrenado!")
        
        return True, f"✅ Equipo 0: {team0_count} | Equipo 1: {team1_count}"
    
    def predict(self, crops):
        """Predice el equipo para nuevos crops"""
        if not self.is_fitted or len(crops) == 0:
            return np.zeros(len(crops), dtype=int)
        
        embeddings = self.extract_embeddings(crops, batch_size=16)
        projections = self.reducer.transform(embeddings)
        predictions = self.kmeans.predict(projections)
        
        return predictions


# --- FUNCIÓN PRINCIPAL DE PROCESAMIENTO ---
def process_video(video_path, model_vertex, model_players, model_ball, 
                  team_classifier, stride=30, progress_bar=None, status_text=None):
    """Procesa el video y retorna el path del video anotado"""
    
    # Fase 1: Recolectar crops
    if status_text:
        status_text.text("📸 Fase 1/3: Recolectando crops de jugadores...")
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_to_sample = frame_count // stride
    
    crops_for_training = []
    
    for i in range(total_frames_to_sample):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * stride)
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model_players.predict(frame, conf=0.3, verbose=False)[0]
        
        for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
            cls_id = int(cls_id.item())
            
            if cls_id == PLAYER_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                crop = frame[y1:y2, x1:x2]
                
                if crop.shape[0] > 30 and crop.shape[1] > 20:
                    crops_for_training.append(crop)
        
        if progress_bar:
            progress_bar.progress((i + 1) / total_frames_to_sample / 3)
    
    cap.release()
    
    if len(crops_for_training) < 20:
        return None, f"❌ Solo se encontraron {len(crops_for_training)} crops. Se necesitan al menos 20."
    
    # Fase 2: Entrenar clasificador
    if status_text:
        status_text.text(f"🤖 Fase 2/3: Entrenando clasificador con {len(crops_for_training)} crops...")
    
    def update_progress(pct, msg):
        if progress_bar:
            progress_bar.progress(0.33 + (pct * 0.33))
        if status_text:
            status_text.text(f"🤖 Fase 2/3: {msg}")
    
    success, message = team_classifier.fit(crops_for_training, progress_callback=update_progress)
    
    if not success:
        return None, message
    
    # Fase 3: Procesar video completo
    if status_text:
        status_text.text("🎬 Fase 3/3: Procesando video completo...")
    
    ball_tracker = BallTracker(buffer_size=30)
    deepsort_tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resize_size = (640, 640)
    
    # Archivo temporal de salida
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))
    
    track_team_map = {}
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated_frame = frame.copy()
        
        # Detectar balón
        ball_results = model_ball.predict(frame, conf=0.3, verbose=False)[0]
        detections_ball = sv.Detections.from_ultralytics(ball_results)
        position = ball_tracker.update(detections_ball)
        
        if position is not None:
            x, y = map(int, position)
            cv2.circle(annotated_frame, (x, y), 10, (0, 255, 0), -1)
        
        # Detectar jugadores
        frame_resized = cv2.resize(frame, resize_size)
        scale_x = orig_width / resize_size[0]
        scale_y = orig_height / resize_size[1]
        
        vertex_results = model_vertex.predict(frame_resized, conf=0.3, verbose=False)[0]
        player_results = model_players.predict(frame_resized, conf=0.3, verbose=False)[0]
        
        # Preparar detecciones para DeepSORT
        detections_for_deepsort = []
        
        for box, cls_id, conf in zip(player_results.boxes.xyxy, 
                                       player_results.boxes.cls,
                                       player_results.boxes.conf):
            x1, y1, x2, y2 = box.cpu().numpy()
            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
            
            w, h = x2 - x1, y2 - y1
            detections_for_deepsort.append(([x1, y1, w, h], conf.item(), int(cls_id.item())))
        
        # Actualizar DeepSORT
        tracks = deepsort_tracker.update_tracks(detections_for_deepsort, frame=frame)
        
        # Recolectar crops de players
        player_tracks = []
        player_crops = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            cls_id = track.det_class if hasattr(track, 'det_class') else 0
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            if cls_id == PLAYER_CLASS_ID and track_id not in track_team_map:
                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] > 30 and crop.shape[1] > 20:
                    player_tracks.append(track_id)
                    player_crops.append(crop)
        
        # Clasificar nuevos players
        if len(player_crops) > 0:
            teams = team_classifier.predict(player_crops)
            for track_id, team in zip(player_tracks, teams):
                track_team_map[track_id] = int(team)
        
        # Dibujar todas las detecciones
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            cls_id = track.det_class if hasattr(track, 'det_class') else 0
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            if cls_id == PLAYER_CLASS_ID:
                team = track_team_map.get(track_id, 0)
                color = TEAM_COLORS[team]
                label = f"ID:{track_id} E{team}"
            elif cls_id == GOALKEEPER_CLASS_ID:
                color = TEAM_COLORS['goalkeeper']
                label = f"ID:{track_id} GK"
            elif cls_id == REFEREE_CLASS_ID:
                color = TEAM_COLORS['referee']
                label = f"ID:{track_id} Ref"
            else:
                color = (255, 255, 255)
                label = f"ID:{track_id}"
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Dibujar vértices
        if vertex_results.keypoints is not None:
            keypoints = vertex_results.keypoints.xy[0].cpu().numpy()
            confidences = vertex_results.keypoints.conf[0].cpu().numpy()
            for (x, y), conf in zip(keypoints, confidences):
                if conf > 0.5:
                    x, y = int(x * scale_x), int(y * scale_y)
                    cv2.circle(annotated_frame, (x, y), 5, (255, 0, 0), -1)
        
        # Estadísticas
        team0_count = sum(1 for t in track_team_map.values() if t == 0)
        team1_count = sum(1 for t in track_team_map.values() if t == 1)
        
        cv2.putText(annotated_frame, f"Equipo 0: {team0_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Equipo 1: {team1_count}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        out.write(annotated_frame)
        
        if progress_bar:
            progress_bar.progress(0.66 + ((frame_idx + 1) / frame_count * 0.34))
    
    cap.release()
    out.release()
    
    team0_final = sum(1 for t in track_team_map.values() if t == 0)
    team1_final = sum(1 for t in track_team_map.values() if t == 1)
    
    return output_path, f"✅ Equipo 0: {team0_final} | Equipo 1: {team1_final}"


# --- INTERFAZ STREAMLIT ---
def main():
    st.title("⚽ Clasificador Automático de Equipos con IA")
    st.markdown("### Detecta y clasifica jugadores en equipos usando Deep Learning")
    
    # Sidebar con configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        stride = st.slider("Frames a saltar (calibración)", 10, 60, 30, 
                          help="Menor valor = más crops pero más lento")
        
        st.markdown("---")
        st.markdown("### 🎨 Colores")
        st.markdown("🔴 **Equipo 0**")
        st.markdown("🔵 **Equipo 1**")
        st.markdown("🟡 **Porteros**")
        st.markdown("⚪ **Árbitros**")
        
        st.markdown("---")
        st.markdown("### 📊 Clases detectadas")
        st.markdown("- Ball (ID: 0)")
        st.markdown("- Goalkeeper (ID: 1)")
        st.markdown("- Player (ID: 2) ⭐")
        st.markdown("- Referee (ID: 3)")
    
    # Cargar modelos (solo una vez)
    model_vertex, model_players, model_ball = load_yolo_models()
    embeddings_model, embeddings_processor, device = load_siglip_model()
    
    st.success(f"✅ Modelos cargados correctamente (Dispositivo: {device})")
    
    # Upload de video
    uploaded_file = st.file_uploader("📹 Sube tu video de fútbol", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Guardar archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Mostrar video original
        st.video(video_path)
        
        # Botón de procesamiento
        if st.button("🚀 Procesar Video", type="primary"):
            # Crear clasificador
            team_classifier = TeamClassifier(embeddings_model, embeddings_processor, device)
            
            # Barras de progreso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Procesar
            output_path, message = process_video(
                video_path, model_vertex, model_players, model_ball,
                team_classifier, stride, progress_bar, status_text
            )
            
            progress_bar.empty()
            
            if output_path:
                status_text.success(message)
                
                st.markdown("---")
                st.subheader("🎬 Video Procesado")
                st.video(output_path)
                
                # Botón de descarga
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="⬇️ Descargar Video",
                        data=f,
                        file_name="video_clasificado.mp4",
                        mime="video/mp4"
                    )
                
                # Limpiar archivo temporal
                os.unlink(output_path)
            else:
                status_text.error(message)
        
        # Limpiar archivo temporal del upload
        os.unlink(video_path)

if __name__ == "__main__":
    main()
