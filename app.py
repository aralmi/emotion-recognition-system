import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import pandas as pd
import time
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode 
import av
from io import BytesIO
import random
import uuid
import shutil
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime

# Page Config 
st.set_page_config(page_title="Распознавание Эмоций", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
def load_css():
    st.markdown("""
    <style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    [data-testid="column"] {
        padding-left: 0.25rem !important;
        padding-right: 0.25rem !important;
    }
    .main .block-container {
        max-width: 1400px;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Constants
OUR_CNN_MODEL_PATH = "models/emotion_model_our_architecture.pth"
VGG16_MODEL_PATH = "models/emotion_model_vgg16.pth"
EMOTION_LABELS = {0: 'Гнев', 1: 'Отвращение', 2: 'Страх', 3: 'Радость', 4: 'Грусть', 5: 'Удивление', 6: 'Нейтральная'}
EMOTION_COLORS = { 'Гнев': '#EF4444', 'Отвращение': '#8B5CF6', 'Страх': '#6B7280', 'Радость': '#22C55E', 'Грусть': '#3B82F6', 'Удивление': '#F97316', 'Нейтральная': '#FBBF24' }
EMOTION_TO_ENGLISH = {
    'Гнев': 'Anger',
    'Отвращение': 'Disgust',
    'Страх': 'Fear',
    'Радость': 'Happy',
    'Грусть': 'Sad',
    'Удивление': 'Surprise',
    'Нейтральная': 'Neutral'
}
IMG_SIZE = 48
ANALYSIS_IMG_DIR = ".quality_analysis_images"
METADATA_CSV_PATH = os.path.join(ANALYSIS_IMG_DIR, "metadata.csv")
COMPARISON_HISTORY_PATH = os.path.join(ANALYSIS_IMG_DIR, "comparison_history.csv")

# Session State Initialization 
def initialize_session_state():
    # ОЧИСТКА: Удаляем всю папку, кроме файла истории
    COMPARISON_HISTORY_FILE = os.path.join(ANALYSIS_IMG_DIR, "comparison_history.csv")

    history_exists = os.path.exists(COMPARISON_HISTORY_FILE)
    history_backup = None

    if history_exists:
        try:
            history_backup = pd.read_csv(COMPARISON_HISTORY_FILE)
        except:
            history_backup = None

    if os.path.exists(ANALYSIS_IMG_DIR):
        shutil.rmtree(ANALYSIS_IMG_DIR)

    os.makedirs(ANALYSIS_IMG_DIR, exist_ok=True)

    if history_backup is not None:
        history_backup.to_csv(COMPARISON_HISTORY_FILE, index=False)
        
    defaults = {
        'setup_complete': False, 'image_to_process': None, 'session_history': [], 'total_processed': 0,
        'emotion_counts': {label: 0 for label in EMOTION_LABELS.values()}, 'selected_model': "CNN",
        'ground_truth_labels': {}, 'run_comparison_analysis': False, 'temp_analysis_entries': [],
        'show_history': False
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

# Model Definitions & Loading 
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1=nn.Conv2d(1,32,3,1,1); self.bn1=nn.BatchNorm2d(32); self.pool1=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(32,64,3,1,1); self.bn2=nn.BatchNorm2d(64); self.pool2=nn.MaxPool2d(2,2)
        self.conv3=nn.Conv2d(64,128,3,1,1); self.bn3=nn.BatchNorm2d(128); self.pool3=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(128*6*6,256); self.dropout1=nn.Dropout(0.5)
        self.fc2=nn.Linear(256,7); self.relu=nn.ReLU()
    def forward(self,x):
        x=self.relu(self.bn1(self.conv1(x))); x=self.pool1(x); x=self.relu(self.bn2(self.conv2(x))); x=self.pool2(x)
        x=self.relu(self.bn3(self.conv3(x))); x=self.pool3(x); x=x.view(x.size(0),-1)
        x=self.relu(self.fc1(x)); x=self.dropout1(x); return self.fc2(x)

@st.cache_resource
def load_all_models():
    models_cache = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        cnn_model = EmotionCNN().to(device)
        cnn_model.load_state_dict(torch.load(OUR_CNN_MODEL_PATH, map_location=device))
        cnn_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((48, 48)), transforms.Grayscale(1), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        models_cache["CNN"] = {"model": cnn_model.eval(), "transform": cnn_transform}
    except Exception as e:
        st.error(f"Не удалось загрузить 'CNN': {e}")

    try:
        vgg_model = models.vgg16(weights=None)
        vgg_model.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 7))
        vgg_model.load_state_dict(torch.load(VGG16_MODEL_PATH, map_location=device))
        vgg_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((48, 48)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        models_cache["VGG16"] = {"model": vgg_model.to(device).eval(), "transform": vgg_transform}
    except Exception as e:
        st.error(f"Не удалось загрузить 'VGG16': {e}")
        
    return models_cache, device

# --- Other Helper Functions ---
@st.cache_resource
def load_face_detector(): return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def setup_example_images():
    if st.session_state.setup_complete: return True
    with st.spinner("Создание примеров изображений (это может занять до минуты)..."):
        try:
            if not os.path.exists("fer2013.csv"): st.error("'fer2013.csv' не найден."); st.session_state.setup_complete = True; return False
            base_path = "assets"
            if os.path.exists(base_path): shutil.rmtree(base_path)
            os.makedirs(base_path)
            
            df = pd.read_csv("fer2013.csv")
            num_examples = 10
            
            for model_tag in ["cnn", "vgg"]:
                model_base_path = os.path.join(base_path, model_tag)
                saved_counts = {name: 0 for name in EMOTION_LABELS.values()}
                for _, row in df.iterrows():
                    emotion_name = EMOTION_LABELS.get(row['emotion'])
                    if emotion_name and saved_counts[emotion_name] < num_examples:
                        pixels = np.array(row['pixels'].split(), 'uint8').reshape((48, 48))
                        if model_tag == "vgg":
                            pixels = cv2.resize(pixels, (224, 224))
                            pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
                        emotion_dir = os.path.join(model_base_path, emotion_name)
                        os.makedirs(emotion_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(emotion_dir, f"{saved_counts[emotion_name]}.png"), pixels)
                        saved_counts[emotion_name] += 1
                    if all(count >= num_examples for count in saved_counts.values()): break
            st.success("Примеры изображений созданы.")
            st.session_state.setup_complete = True
            time.sleep(1); st.rerun()
        except Exception as e: st.error(f"Ошибка при создании примеров: {e}"); return False
    return True

def predict_emotion(img_np, device, model, transform, filename, update_stats=True, use_face_detection=True):
    face_detector = load_face_detector()
    
    if use_face_detection:
        gray_image = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_image, 1.1, 5, minSize=(30, 30))
        if not len(faces):
            return None, "Лицо не найдено"
        face_rois = [(img_np[y:y+h, x:x+w], (x, y, w, h)) for (x, y, w, h) in faces]
    else:
        if img_np is None or img_np.size == 0:
            return None, "Нет данных для обработки"
        face_rois = [(img_np, (0, 0, img_np.shape[1], img_np.shape[0]))]
    
    if not face_rois:
        return None, "Нет данных для обработки"
    
    if update_stats: st.session_state.total_processed += len(face_rois)
    results = []
    for face_roi, box in face_rois:
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_tensor = transform(face_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(face_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            conf, idx = torch.max(probs, 0); emotion = EMOTION_LABELS[idx.item()]
        if update_stats:
            st.session_state.emotion_counts[emotion] += 1
            st.session_state.session_history.insert(0, {"Время": pd.Timestamp.now().strftime("%H:%M:%S"), "Файл": filename, "Эмоция": emotion, "Точность (%)": f"{conf.item()*100:.2f}"})
            st.session_state.session_history = st.session_state.session_history[:30]
        results.append({"box": box, "emotion": emotion, "confidence": conf.item()*100, "probabilities": {EMOTION_LABELS[i]:p.item()*100 for i,p in enumerate(probs)}, "face_roi": face_roi})
    return results, None

# Webcam Transformer 
class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        try:
            self.models_cache, self.device = load_all_models()
            self.face_detector = load_face_detector()
            print(f"Модели загружены. Device: {self.device}")
        except Exception as e:
            print(f"Ошибка загрузки моделей в __init__: {e}")
            self.models_cache = {}
            self.device = "cpu"
            
    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        try:
            img = frame.to_ndarray(format="bgr24")
            
            if not self.models_cache:
                print("Модели не загружены, возвращаем оригинальный кадр")
                return img
            
            model_name = st.session_state.get('selected_model', "CNN")
            model_pack = self.models_cache.get(model_name)
            
            if not model_pack:
                print(f"Модель '{model_name}' не найдена в кэше. Доступные: {list(self.models_cache.keys())}")
                return img
            
            try:
                results, error = predict_emotion(
                    img, self.device, model_pack["model"], model_pack["transform"], "webcam", 
                    update_stats=False, use_face_detection=True
                )
                
                if error:
                    print(f"Ошибка в predict_emotion: {error}")
                    return img
                
                if results:
                    for res in results:
                        (x, y, w, h) = res['box']
                        emotion = res['emotion']
                        conf = res['confidence']
                        
                        hex_color = EMOTION_COLORS.get(emotion, '#808080')
                        color_bgr = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))[::-1]
                        
                        cv2.rectangle(img, (x, y), (x+w, y+h), color_bgr, 2)
                        
                        emotion_display = EMOTION_TO_ENGLISH.get(emotion, emotion)
                        text = f"{emotion_display} ({conf:.1f}%)"
                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(img, (x, y-th-10), (x+tw, y), color_bgr, -1)
                        cv2.putText(img, text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    print("Лиц не обнаружено на текущем кадре")
                    
            except Exception as e:
                print(f"Ошибка при обработке кадра: {e}")
                return img
            return img
        except Exception as e:
            print(f"Критическая ошибка в transform(): {e}")
            return frame.to_ndarray(format="bgr24")

# --- Page Handlers ---
def handle_single_photo(models_cache, device):
    st.header("Анализ фотографии")
    st.info("""
    **Анализ одиночных фотографий**
    
    Загрузите фотографию с лицом, и система определит эмоцию выбранной моделью:
    - **CNN** - собственная нейросеть
    - **VGG16** - предобученная модель
    """)
    with st.container(border=True):
        uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png", "bmp"])
        st.markdown("---")
        if st.button("Загрузить пример"):
            try:
                model_folder = "cnn" if st.session_state.selected_model == "CNN" else "vgg"
                example_base_path = os.path.join("assets", model_folder)
                asset_dirs = [d for d in os.listdir(example_base_path) if os.path.isdir(os.path.join(example_base_path, d)) and os.listdir(os.path.join(example_base_path, d))]
                if not asset_dirs: st.error("Нет примеров."); return
                random_emotion = random.choice(asset_dirs)
                img_path = os.path.join(example_base_path, random_emotion, random.choice(os.listdir(os.path.join(example_base_path, random_emotion))))
                with open(img_path, "rb") as f:
                    st.session_state.image_to_process = {"name": f"Пример: {random_emotion}", "data": f.read(), "is_example": True}
                st.rerun()
            except (FileNotFoundError, IndexError): st.error("Ошибка при загрузке примера.")
            
    image_source = uploaded_file or st.session_state.image_to_process
    if image_source:
        is_example = isinstance(image_source, dict) and image_source.get("is_example")
        name = image_source.name if hasattr(image_source, 'name') else image_source.get("name", "Пример")
        data = image_source.getvalue() if hasattr(image_source, 'getvalue') else image_source.get("data")
        st.session_state.image_to_process = None
        img_np = np.array(Image.open(BytesIO(data)).convert("RGB")); img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        model_pack = models_cache.get(st.session_state.selected_model)
        if not model_pack: st.error(f"Модель '{st.session_state.selected_model}' не загружена."); return
        
        results, error = predict_emotion(img_cv, device, model_pack["model"], model_pack["transform"], name, update_stats=not is_example, use_face_detection=not is_example)
        
        if results and not error:
            unique_id = f"temp_{uuid.uuid4().hex[:6]}"
            st.session_state.temp_analysis_entries.append({
                "unique_id": unique_id, "image": results[0]['face_roi'],
                "predicted_emotion": results[0]['emotion'], "model_name": st.session_state.selected_model,
                "ground_truth": 'Не размечено', "is_example": is_example
            })
        if error: st.error(error)
        elif results:
            with st.container(border=True):
                res = results[0]; st.subheader(f"Результат")
                col1, col2 = st.columns([1, 1.5])
                with col1:
                    img_display = img_cv.copy(); (x, y, w, h) = res["box"]; color_hex = EMOTION_COLORS.get(res['emotion'], '#808080'); color_bgr = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0,2,4))[::-1]
                    if not is_example: cv2.rectangle(img_display, (x, y), (x+w, y+h), color_bgr, 3)
                    st.image(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col2:
                    st.markdown(f"#### Эмоция: <span style='color:{color_hex};'>{res['emotion']}</span>", unsafe_allow_html=True); st.markdown(f"**Точность:** `{res['confidence']:.2f}%`")
                    probs_df = pd.DataFrame([res['probabilities']]).T.reset_index(); probs_df.columns = ["Эмоция", "Вероятность (%)"]
                    fig, ax = plt.subplots(); sns.barplot(x="Вероятность (%)", y="Эмоция", data=probs_df.sort_values("Вероятность (%)", ascending=False), ax=ax, palette={e: EMOTION_COLORS.get(e, '#808080') for e in probs_df['Эмоция']}); ax.set_xlim(0, 100); st.pyplot(fig)

def handle_batch_processing(models_cache, device):
    st.header("Массовая обработка")
    st.info("""
    **Обработка нескольких фотографий**
    
    Загрузите несколько файлов (JPG, PNG) для одновременного анализа.
    """)
    with st.container(border=True):
        files = st.file_uploader("Выберите несколько изображений", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)
        
        if files and st.button("Начать обработку"):
            model_pack = models_cache.get(st.session_state.selected_model)
            if not model_pack:
                st.error(f"Модель '{st.session_state.selected_model}' не загружена.")
                return

            gallery_results = []
            batch_emotion_counts = {label: 0 for label in EMOTION_LABELS.values()}
            success_count = 0; error_count = 0
            
            bar = st.progress(0, "Инициализация...")
            model_name = st.session_state.selected_model
            
            for i, file in enumerate(files):
                bar.progress((i + 1) / len(files), f"Обработка: {file.name}")
                try:
                    img_np = np.array(Image.open(file).convert("RGB"))
                    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    
                    results, error = predict_emotion(img_cv, device, model_pack["model"], model_pack["transform"], file.name)
                    
                    if error or not results: error_count += 1
                    else:
                        success_count += 1
                        for res in results:
                            unique_id = f"temp_{uuid.uuid4().hex[:6]}"
                            st.session_state.temp_analysis_entries.append({
                                "unique_id": unique_id, "image": res['face_roi'],
                                "predicted_emotion": res['emotion'], "model_name": model_name,
                                "ground_truth": 'Не размечено', "is_example": False
                            })
                            batch_emotion_counts[res['emotion']] += 1
                            gallery_results.append({**res, 'filename': file.name, 'original_image': img_cv.copy()})

                except Exception as e:
                    error_count += 1; st.toast(f"Ошибка при обработке {file.name}: {e}")

            st.markdown("---")
            st.subheader("Результаты")

            col1, col2 = st.columns(2)
            with col1: st.metric("Успешно обработано", f"{success_count} / {len(files)}")
            with col2: st.metric("Ошибки (лицо не найдено или др.)", f"{error_count}")

            st.subheader("Распределение эмоций в этой пачке")
            total_emotions_in_batch = sum(batch_emotion_counts.values())
            if total_emotions_in_batch > 0:
                for emotion, count in sorted(batch_emotion_counts.items(), key=lambda item: item[1], reverse=True):
                    if count > 0:
                        percentage = (count / total_emotions_in_batch) * 100
                        st.markdown(f"**{emotion}**"); st.progress(int(percentage), text=f"{percentage:.1f}% ({count})")
            else: st.info("В этой пачке не было найдено ни одной эмоции.")
            
            st.markdown("---")
            st.subheader("Галерея результатов")
            
            if not gallery_results:
                st.warning("Не удалось обработать ни одного лица для отображения в галерее.")
                return

            cols = st.columns(4)
            for idx, res in enumerate(gallery_results):
                with cols[idx % 4]:
                    img_display = res['original_image'].copy()
                    (x, y, w, h) = res["box"]
                    color_hex = EMOTION_COLORS.get(res['emotion'], '#808080')
                    color_bgr = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0,2,4))[::-1]
                    cv2.rectangle(img_display, (x, y), (x+w, y+h), color_bgr, 3)
                    
                    st.image(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.markdown(f"<h5 style='text-align: center; color: {color_hex};'>{res['emotion']}</h5>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center;'>Точность: {res['confidence']:.1f}%</p>", unsafe_allow_html=True)
                    st.caption(f"Файл: {res['filename']}")

def handle_session_stats():
    st.header("Статистика сессии")
    st.info("""
    **Статистика текущей сессии**
    
    Сводка по всем обработанным в этой сессии фотографиям.
    """)
    with st.container(border=True):
        if not st.session_state.session_history: st.info("В этой сессии нет данных."); return
        st.metric("Всего обработано за сессию", st.session_state.total_processed)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Распределение по эмоциям"); df = pd.DataFrame(st.session_state.emotion_counts.items(), columns=["Эмоция", "Количество"])
            st.dataframe(df[df['Количество'] > 0].sort_values("Количество", ascending=False), use_container_width=True)
        with col2:
            st.subheader("Распределение эмоций"); found = df[df['Количество'] > 0]
            if not found.empty:
                fig, ax = plt.subplots(); sns.barplot(data=found, x="Количество", y="Эмоция", ax=ax, palette=[EMOTION_COLORS.get(e, 'lightgray') for e in found['Эмоция']]); ax.set_ylabel(""); ax.set_xlabel("Количество"); st.pyplot(fig)
        st.markdown("---"); st.subheader("История распознаваний"); st.dataframe(pd.DataFrame(st.session_state.session_history), use_container_width=True)

def display_analysis_results(df, model_name_str):
    analyzable_df = df[df['predicted_emotion'] != 'Не определена'].reset_index(drop=True)
    if analyzable_df.empty: st.warning(f"Для модели '{model_name_str}' не найдено размеченных данных с успешными предсказаниями."); return
    true_labels, pred_labels = analyzable_df['ground_truth'], analyzable_df['predicted_emotion']
    st.metric(f"Accuracy ({model_name_str})", f"{accuracy_score(true_labels, pred_labels):.2%}")
    labels = sorted(list(set(true_labels) | set(pred_labels)))
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Матрица ошибок"); cm = confusion_matrix(true_labels, pred_labels, labels=labels)
        fig, ax = plt.subplots(figsize=(7, 5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_ylabel('Истинная эмоция'); ax.set_xlabel('Предсказанная эмоция'); plt.xticks(rotation=45); plt.yticks(rotation=0); st.pyplot(fig)
    with col2:
        st.subheader("Отчет по метрикам"); report = classification_report(true_labels, pred_labels, labels=labels, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose().style.format({'precision':'{:.2f}', 'recall':'{:.2f}', 'f1-score':'{:.2f}'}))

def save_comparison_to_history(comparison_table):
    total_images = len(comparison_table)
    if total_images == 0:
        st.error("Нечего сохранять - нет данных сравнения")
        return

    cnn_correct = sum(comparison_table['CNN правильно'])
    vgg_correct = sum(comparison_table['VGG16 правильно'])

    session_record = {
        'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"), 'total_images': total_images,
        'cnn_correct': cnn_correct, 'vgg_correct': vgg_correct,
        'cnn_accuracy': f"{100*cnn_correct/total_images:.1f}%", 'vgg_accuracy': f"{100*vgg_correct/total_images:.1f}%",
        'difference_vgg_minus_cnn': f"{100*(vgg_correct-cnn_correct)/total_images:+.1f}%"
    }
    new_record_df = pd.DataFrame([session_record])
    history_path = COMPARISON_HISTORY_PATH

    try:
        if os.path.exists(history_path):
            history_df = pd.read_csv(history_path)
            history_df = pd.concat([history_df, new_record_df], ignore_index=True)
        else:
            history_df = new_record_df

        os.makedirs(ANALYSIS_IMG_DIR, exist_ok=True)
        history_df.to_csv(history_path, index=False)
        st.success("Результаты сохранены в историю сравнений!")
    except Exception as e:
        st.error(f"Ошибка при сохранении в историю: {e}")

def show_comparison_history():

    history_path = COMPARISON_HISTORY_PATH

    if not os.path.exists(history_path):

        st.info("История отсутствует. Выполните несколько сравнений, чтобы увидеть тренды.")

        return

    try:

        history_df = pd.read_csv(history_path)

        if history_df.empty:

            st.info("История отсутствует. Выполните несколько сравнений, чтобы увидеть тренды.")

            return

    except (FileNotFoundError, pd.errors.EmptyDataError):

        st.info("История отсутствует или файл истории поврежден.")

        return



    # --- Display overall statistics ---

    try:

        st.markdown("---")

        st.subheader("Общая статистика по всем сессиям")

        

        required_cols = ['total_images', 'cnn_accuracy', 'vgg_accuracy']

        if not all(col in history_df.columns for col in required_cols):

             raise KeyError("Отсутствуют необходимые колонки для расчета общей статистики.")



        cnn_accs_numeric = history_df['cnn_accuracy'].str.rstrip('%').astype(float)

        vgg_accs_numeric = history_df['vgg_accuracy'].str.rstrip('%').astype(float)

        

        total_images = history_df['total_images'].sum()

        avg_cnn = (cnn_accs_numeric * history_df['total_images']).sum() / total_images if total_images > 0 else 0

        avg_vgg = (vgg_accs_numeric * history_df['total_images']).sum() / total_images if total_images > 0 else 0

        avg_diff = avg_vgg - avg_cnn

        

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric("Всего изображений", int(total_images))

        with col2:

            st.metric("CNN (общая точность)", f"{avg_cnn:.1f}%")

        with col3:

            st.metric("VGG16 (общая точность)", f"{avg_vgg:.1f}%")

        with col4:

            st.metric("Разница VGG16 - CNN", f"{avg_diff:+.1f}%")

    except (KeyError, ValueError, TypeError):

        st.warning("Не удалось рассчитать общую статистику. Возможно, файл истории имеет старый формат.", icon="⚠️")





    # --- Display Table of all sessions ---

    try:

        st.markdown("---")

        st.subheader("Таблица всех сессий")

        

        base_cols_to_display = ['session_id', 'total_images', 'cnn_accuracy', 'vgg_accuracy', 'difference_vgg_minus_cnn']

        cols_that_exist = [col for col in base_cols_to_display if col in history_df.columns]

        display_df = history_df[cols_that_exist].copy()

        

        rename_map = {

            'session_id': 'Дата/Время', 'total_images': 'Фото', 'cnn_accuracy': 'CNN',

            'vgg_accuracy': 'VGG16', 'difference_vgg_minus_cnn': 'Разница (VGG-CNN)'

        }

        display_df.rename(columns={k: v for k, v in rename_map.items() if k in cols_that_exist}, inplace=True)



        if not display_df.empty:

            st.dataframe(display_df, use_container_width=True)

    except Exception as e:

        st.error(f"Не удалось отобразить таблицу сессий: {e}")

def handle_quality_analysis(device, models_cache):
    st.header("Проверка качества")
    st.info("""
    **Проверка качества моделей**
    
    1. **Загрузить фото** - добавьте фотографии
    2. **Разметить эмоции** - выберите правильную эмоцию для каждой
    3. **Сохранить** - нажмите "Сохранить разметку"
    4. **Проанализировать модель** - смотрите результаты CNN или VGG16
    5. **Сравнить** - нажмите "Сравнение моделей" чтобы сравнить обе
    6. **История** - кнопка "История сравнений" показывает результаты всех сессий
    
    **Результаты:**
    - Матрица ошибок (какие эмоции путает)
    - Accuracy, Precision, Recall, F1-score
    - Отчет по каждой эмоции
    - Общая статистика всех сессий
    """)
    
    if not os.path.exists(ANALYSIS_IMG_DIR):
        os.makedirs(ANALYSIS_IMG_DIR)
    
    try:
        persistent_df = pd.read_csv(METADATA_CSV_PATH)
        for col in ['model_name', 'ground_truth']:
            if col not in persistent_df.columns: persistent_df[col] = 'Неизвестно'
        persistent_df = persistent_df.reset_index(drop=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        persistent_df = pd.DataFrame(columns=['unique_id', 'original_filename', 'predicted_emotion', 'model_name', 'ground_truth'])

    temp_entries_for_df = []
    for item in st.session_state.temp_analysis_entries:
        is_example_item = item.get('is_example', False)
        temp_filepath = os.path.join(ANALYSIS_IMG_DIR, f"{item['unique_id']}.png")
        if not os.path.exists(temp_filepath):
            cv2.imwrite(temp_filepath, item['image'])
        
        if not is_example_item:
            temp_entries_for_df.append({
                'unique_id': item['unique_id'], 'original_filename': 'Загруженное фото',
                'predicted_emotion': item['predicted_emotion'], 'model_name': item['model_name'],
                'ground_truth': st.session_state.ground_truth_labels.get(item['unique_id'], 'Не размечено')
            })

    if temp_entries_for_df:
        temp_df = pd.DataFrame(temp_entries_for_df)
        combined_df = pd.concat([persistent_df, temp_df], ignore_index=True).reset_index(drop=True)
    else:
        combined_df = persistent_df.copy()

    # Убедитесь что combined_df имеет нужные колонки даже если пусто
    if combined_df.empty:
        combined_df = pd.DataFrame(columns=['unique_id', 'original_filename', 'predicted_emotion', 'model_name', 'ground_truth'])

    st.session_state.ground_truth_labels = pd.Series(combined_df['ground_truth'].values, 
                                                 index=combined_df['unique_id']).to_dict() if not combined_df.empty else {}

    st.subheader("Выбор эмоций")
    if combined_df.empty: st.info("Нет фотографий.")
    else:
        for _, row in combined_df.sort_values('unique_id', ascending=False).iterrows():
            img_path = os.path.join(ANALYSIS_IMG_DIR, f"{row['unique_id']}.png")
            if os.path.exists(img_path):
                col1, col2, col3 = st.columns([1, 2, 2])
                with col1: st.image(img_path, width=100)
                with col2: st.write(f"**Предсказание ({row['model_name']}):**"); st.markdown(f"<span style='color:{EMOTION_COLORS.get(row['predicted_emotion'])};'>{row['predicted_emotion']}</span>", unsafe_allow_html=True)
                with col3:
                    options = ['Не размечено'] + list(EMOTION_LABELS.values()); current_label = st.session_state.ground_truth_labels.get(row['unique_id'], 'Не размечено')
                    new_label = st.selectbox("Укажите эмоцию:", options, index=options.index(current_label) if current_label in options else 0, key=f"label_{row['unique_id']}")
                    if new_label != current_label: st.session_state.ground_truth_labels[row['unique_id']] = new_label; st.rerun()

    if st.button("Сохранить разметку"):
        combined_df['ground_truth'] = combined_df['unique_id'].map(st.session_state.ground_truth_labels).fillna('Не размечено')
        combined_df.to_csv(METADATA_CSV_PATH, index=False)
        st.success("Разметка для загруженных файлов сохранена!")

    st.markdown("---")
    st.subheader("Результаты анализа")
    
    col_history = st.columns(1)[0]
    with col_history:
        if st.button("История сравнений всех сессий"):
            st.session_state.show_history = not st.session_state.show_history # Toggle visibility
    
    # NEW: Показываем историю если она запрошена
    if st.session_state.get('show_history', False):
        show_comparison_history()

    # ИСПРАВЛЕНИЕ: безопасное создание labeled_df
    if not combined_df.empty and combined_df['unique_id'].isin(st.session_state.ground_truth_labels.keys()).any():
        labeled_df = combined_df[
            combined_df['unique_id'].map(st.session_state.ground_truth_labels).fillna('Не размечено').apply(
                lambda x: x != 'Не размечено'
            )
        ].copy().reset_index(drop=True)
    else:
        labeled_df = pd.DataFrame(columns=['unique_id', 'original_filename', 'predicted_emotion', 'model_name', 'ground_truth'])
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Проанализировать текущую модель"): st.session_state.run_comparison_analysis = False
    with col2:
        if st.button("Сравнение моделей"): st.session_state.run_comparison_analysis = True

    if not st.session_state.run_comparison_analysis:
        # Defensive check: Ensure 'model_name' column exists in labeled_df
        if 'model_name' not in labeled_df.columns:
            st.error("Ошибка: В DataFrame 'labeled_df' отсутствует колонка 'model_name'. Проверьте инициализацию данных.")
            # Create an empty DataFrame with the expected columns to prevent further errors
            model_specific_df = pd.DataFrame(columns=['unique_id', 'original_filename', 'predicted_emotion', 'model_name', 'ground_truth'])
        else:
            model_specific_df = labeled_df[labeled_df['model_name'] == st.session_state.selected_model].reset_index(drop=True)
        if not model_specific_df.empty: 
            display_analysis_results(model_specific_df, st.session_state.selected_model)
        else: 
            st.warning(f"Нет размеченных данных для анализа модели '{st.session_state.selected_model}'.")
    else:
        if labeled_df.empty: 
            st.error("Нет размеченных изображений для сравнения.")
            return
        
        st.markdown("---")
        st.subheader("Сравнение моделей")
        st.info(f"Обе модели будут оценены на {len(labeled_df)} размеченных изображениях")
        
        comparison_df = labeled_df.copy().reset_index(drop=True)
        
        temp_images_cache = {item['unique_id']: item['image'] for item in st.session_state.temp_analysis_entries}
        
        st.write("### Результаты для `CNN`")
        cnn_df = comparison_df.copy()
        cnn_preds = []
        
        progress_bar = st.progress(0, "Обработка изображений CNN...")
        for idx, row in cnn_df.iterrows():
            unique_id = row.unique_id
            img = None
            
            # Сначала проверяем кэш
            if unique_id in temp_images_cache:
                img = temp_images_cache[unique_id]
            else:
                # Если нет в кэше, загружаем с диска
                img_path = os.path.join(ANALYSIS_IMG_DIR, f"{unique_id}.png")
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)

            if img is not None:
                try:
                    results, _ = predict_emotion(img, device, models_cache["CNN"]["model"], models_cache["CNN"]["transform"], filename="", update_stats=False, use_face_detection=False)
                    cnn_preds.append(results[0]['emotion'] if results else "Не определена")
                except Exception as e:
                    print(f"Ошибка CNN: {e}"); cnn_preds.append("Не определена")
            else:
                cnn_preds.append("Не определена")
            
            progress_bar.progress((idx + 1) / len(cnn_df), f"CNN: обработано {idx + 1}/{len(cnn_df)}")
        
        cnn_df['predicted_emotion'] = cnn_preds
        display_analysis_results(cnn_df, "CNN")
        
        st.markdown("---")
        st.write("### Результаты для `VGG16`")
        vgg_df = comparison_df.copy()
        vgg_preds = []
        
        progress_bar = st.progress(0, "Обработка изображений VGG16...")
        for idx, row in vgg_df.iterrows():
            unique_id = row.unique_id
            img = None
            
            # Сначала проверяем кэш
            if unique_id in temp_images_cache:
                img = temp_images_cache[unique_id]
            else:
                # Если нет в кэше, загружаем с диска
                img_path = os.path.join(ANALYSIS_IMG_DIR, f"{unique_id}.png")
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)

            if img is not None:
                try:
                    results, _ = predict_emotion(img, device, models_cache["VGG16"]["model"], models_cache["VGG16"]["transform"], filename="", update_stats=False, use_face_detection=False)
                    vgg_preds.append(results[0]['emotion'] if results else "Не определена")
                except Exception as e:
                    print(f"Ошибка VGG16: {e}"); vgg_preds.append("Не определена")
            else:
                vgg_preds.append("Не определена")
            
            progress_bar.progress((idx + 1) / len(vgg_df), f"VGG16: обработано {idx + 1}/{len(vgg_df)}")
        
        vgg_df['predicted_emotion'] = vgg_preds
        display_analysis_results(vgg_df, "VGG16")
        
        st.markdown("---")
        st.subheader("Таблица результатов")
        comparison_table = pd.DataFrame({
            'Фото': comparison_df['unique_id'], 'Истинная эмоция': comparison_df['ground_truth'],
            'CNN предсказание': cnn_preds, 'VGG16 предсказание': vgg_preds,
            'CNN правильно': [cnn == truth for cnn, truth in zip(cnn_preds, comparison_df['ground_truth'])],
            'VGG16 правильно': [vgg == truth for vgg, truth in zip(vgg_preds, comparison_df['ground_truth'])]
        })
        st.dataframe(comparison_table, use_container_width=True)
        
        cnn_correct = sum(comparison_table['CNN правильно'])
        vgg_correct = sum(comparison_table['VGG16 правильно'])
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("CNN Accuracy", f"{cnn_correct}/{len(comparison_df)} ({100*cnn_correct/len(comparison_df):.1f}%)")
        with col2: st.metric("VGG16 Accuracy", f"{vgg_correct}/{len(comparison_df)} ({100*vgg_correct/len(comparison_df):.1f}%)")
        with col3:
            diff = vgg_correct - cnn_correct
            st.metric("Разница (VGG16 - CNN)", f"{diff:+d} ({100*diff/len(comparison_df):+.1f}%)")
        
        st.markdown("---")
        if st.button("Сохранить результаты"):
            save_comparison_to_history(comparison_table)

def handle_webcam():
    st.header("Камера")
    st.info("""
    **Распознавание эмоций в реальном времени**
    
    Включите веб-камеру для анализа лица в прямом эфире.
    Система определит эмоцию на кадрах с камеры.
    """)
    st.info("Нажмите на кнопку START, чтобы включить камеру. Производительность может быть низкой на CPU, особенно с моделью VGG16.")
    webrtc_streamer(key="emotion_detection", mode=WebRtcMode.SENDRECV, video_processor_factory=EmotionTransformer, media_stream_constraints={"video": True, "audio": False}, async_processing=True)

def add_footer():
    st.markdown("---")
    st.caption("Приложение для распознавания эмоций. Версия 1.0")

# Main App Logic 
def main():
    initialize_session_state(); load_css()
    if not st.session_state.setup_complete:
        setup_example_images()
        
    st.sidebar.title("Навигация")
    st.sidebar.subheader("Выбор модели")
    st.sidebar.radio("Выберите модель:", ("CNN", "VGG16"), key="selected_model", help="Эта модель будет использоваться для всех операций.")
    page = st.sidebar.radio("Выберите раздел:", ["Анализ", "Камера", "Обработка", "Статистика", "Качество"])
    st.sidebar.markdown("---"); st.sidebar.info("Приложение для распознавания эмоций.")
    
    models_cache, device = load_all_models()
    
    if models_cache:
        if page == "Анализ": handle_single_photo(models_cache, device)
        elif page == "Камера": handle_webcam()
        elif page == "Обработка": handle_batch_processing(models_cache, device)
        elif page == "Статистика": handle_session_stats()
        elif page == "Качество": handle_quality_analysis(device, models_cache)
    else:
        st.error(f"Не удалось загрузить ни одной модели.")
    
    add_footer()

if __name__ == "__main__":
    main()