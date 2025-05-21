import streamlit as st
import torch
from PIL import Image
import tempfile
import os
import cv2
import time
import pandas as pd
from collections import Counter
from datetime import datetime
import pyttsx3
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Smart Object Detector", layout="wide")

@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model = load_model()

def speak_predictions(predictions):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    if predictions:
        unique_objects = sorted(set(predictions))
        spoken_text = "Detected objects are: " + ", ".join(unique_objects)
    else:
        spoken_text = "No objects detected."
    engine.say(spoken_text)
    engine.runAndWait()

def save_to_csv(name, email, purpose, predictions):
    df = pd.DataFrame({
        "Name": [name],
        "Email": [email],
        "Purpose": [purpose],
        "Predictions": [", ".join(predictions)],
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    })
    if os.path.exists("user_predictions.csv"):
        df.to_csv("user_predictions.csv", mode='a', header=False, index=False)
    else:
        df.to_csv("user_predictions.csv", index=False)

def delete_row_by_index(index):
    df = pd.read_csv("user_predictions.csv")
    df.drop(index=index, inplace=True)
    df.to_csv("user_predictions.csv", index=False)

def generate_pdf_report(name, email, purpose, predictions, image_path=None):
    pdf_path = "detection_report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Smart Object Detection Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Name: {name}")
    c.drawString(50, height - 100, f"Email: {email}")
    c.drawString(50, height - 120, f"Purpose: {purpose}")
    c.drawString(50, height - 140, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 170, "Detected Objects:")
    c.setFont("Helvetica", 12)
    if predictions:
        counts = Counter(predictions)
        y = height - 190
        for cls, count in counts.items():
            c.drawString(70, y, f"- {cls}: {count}")
            y -= 20
    else:
        c.drawString(70, height - 190, "No objects detected.")
    if image_path and os.path.exists(image_path):
        try:
            img = Image.open(image_path)
            img.thumbnail((300, 300))
            img.save("resized_preview.jpg")
            c.drawImage(ImageReader("resized_preview.jpg"), 300, height - 400, width=200, preserveAspectRatio=True)
        except:
            c.drawString(50, height - 250, "Preview image could not be loaded.")
    c.save()
    return pdf_path

# Detection History
st.sidebar.markdown("## 🧾 Detection History")
if os.path.exists("user_predictions.csv"):
    df_history = pd.read_csv("user_predictions.csv")
    search_name = st.sidebar.text_input("Search by Name")
    if search_name:
        df_history = df_history[df_history["Name"].str.contains(search_name, case=False)]
    for i, row in df_history.tail(10).iterrows():
        st.sidebar.markdown(f"**{row['Name']}** - {row['Predictions']}  ")
        st.sidebar.write(f"{row['Timestamp']}")
        if st.sidebar.button("🗑️", key=f"delete_{i}"):
            delete_row_by_index(i)
            st.experimental_rerun()
else:
    st.sidebar.write("No detection history yet.")

if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

if not st.session_state.form_submitted:
    st.markdown("## Object Detection System for Visually-Challenged 🧑‍💼\nStep 1: Tell us about yourself")
    with st.form("user_info_form"):
        st.markdown("### 📝 Fill in your details")
        name = st.text_input("Name")
        email = st.text_input("Email")
        purpose = st.text_area("What do you plan to use this for?", max_chars=150)
        submitted = st.form_submit_button("Continue")
        if submitted:
            st.session_state.form_submitted = True
            st.session_state.name = name
            st.session_state.email = email
            st.session_state.purpose = purpose
            st.rerun()

if st.session_state.form_submitted:
    st.markdown(f"## 📤 Step 2: Upload your Image, Video, or Use Webcam")
    st.markdown(f"### 👋 Hello, **{st.session_state.name}**! Let's detect some objects.")
    media_type = st.radio("Select Media Type", ["Image", "Video", "Live Webcam"], horizontal=True)

    if media_type == "Image":
        image_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
        if image_file:
            image = Image.open(image_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            with st.spinner("Predicting..."):
                results = model(image)
                predictions = results.pandas().xyxy[0]['name'].tolist()
                results.render()
                pred_img = Image.fromarray(results.ims[0])
                st.image(pred_img, caption="Prediction Result", use_container_width=True)

            st.markdown("### 📋 Detected Objects:")
            if predictions:
                for cls, count in Counter(predictions).items():
                    st.write(f"- **{cls}**: {count}")
            else:
                st.write("No objects detected.")

            speak_predictions(predictions)

            img_path = "prediction_result.jpg"
            pred_img.save(img_path)
            with open(img_path, "rb") as file:
                st.download_button("📥 Download Prediction Image", file, file_name="prediction_result.jpg")

            save_to_csv(st.session_state.name, st.session_state.email, st.session_state.purpose, predictions)
            pdf_path = generate_pdf_report(
                st.session_state.name,
                st.session_state.email,
                st.session_state.purpose,
                predictions,
                img_path
            )
            with open(pdf_path, "rb") as f:
                st.download_button("📄 Download Full Detection Report (PDF)", f, file_name="detection_report.pdf")

    elif media_type == "Video":
        video_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            st.video(tfile.name)
            stframe = st.empty()
            cap = cv2.VideoCapture(tfile.name)
            all_preds = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                predictions = results.pandas().xyxy[0]['name'].tolist()
                all_preds.extend(predictions)
                results.render()
                rendered_frame = results.ims[0]
                stframe.image(rendered_frame, channels="BGR", use_container_width=True)
                time.sleep(0.03)
            cap.release()
            st.markdown("### 📋 Detected Objects:")
            if all_preds:
                for cls, count in Counter(all_preds).items():
                    st.write(f"- **{cls}**: {count}")
            else:
                st.write("No objects detected.")

            speak_predictions(all_preds)

            save_to_csv(st.session_state.name, st.session_state.email, st.session_state.purpose, all_preds)
            pdf_path = generate_pdf_report(
                st.session_state.name,
                st.session_state.email,
                st.session_state.purpose,
                all_preds
            )
            with open(pdf_path, "rb") as f:
                st.download_button("📄 Download Full Detection Report (PDF)", f, file_name="detection_report.pdf")

    elif media_type == "Live Webcam":
        st.markdown("### 📡 Live Webcam")
        st.warning("Webcam will run for 10 seconds. Then snapshot will be auto-captured.")
        snapshot_img = None
        all_preds = []
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        stframe = st.empty()
        while time.time() - start_time < 10:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break
            results = model(frame)
            predictions = results.pandas().xyxy[0]['name'].tolist()
            all_preds.extend(predictions)
            results.render()
            rendered = results.ims[0]
            stframe.image(rendered, channels="BGR", use_container_width=True)
        cap.release()

        st.success("Snapshot taken after 10 seconds")
        snapshot_img = rendered
        st.image(snapshot_img, caption="Snapshot with Detections", use_container_width=True)

        st.markdown("### 📋 Detected Objects:")
        if all_preds:
            for cls, count in Counter(all_preds).items():
                st.write(f"- **{cls}**: {count}")
        else:
            st.write("No objects detected.")

        speak_predictions(all_preds)
        img_path = "webcam_snapshot.jpg"
        cv2.imwrite(img_path, snapshot_img)
        with open(img_path, "rb") as file:
            st.download_button("📥 Download Snapshot Image", file, file_name="webcam_snapshot.jpg")

        save_to_csv(st.session_state.name, st.session_state.email, st.session_state.purpose, all_preds)
        pdf_path = generate_pdf_report(
            st.session_state.name,
            st.session_state.email,
            st.session_state.purpose,
            all_preds,
            img_path
        )
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Full Detection Report (PDF)", f, file_name="detection_report.pdf")
else:
    st.info("Please fill out the form above to get started.")