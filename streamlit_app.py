import streamlit as st
import yt_dlp
import os
import cv2
import numpy as np
from cvit_prediction import predict, load_cvit, preprocess_frame, df_face, is_video, set_result, store_result
from fpdf import FPDF
import matplotlib.pyplot as plt

def download_youtube_video(url, output_dir="downloads"):
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def analyze_video(video_path, model, num_frames=15, fp16=False):
    result = set_result()
    if is_video(video_path):
        st.write(f"Processing video: {os.path.basename(video_path)}")
        result, accuracy, count, pred = predict(
            video_path,
            model,
            fp16,
            result,
            num_frames,
            None,
            "uncategorized",
            0,
        )
        st.write(f"\nPrediction: {pred[1]} {real_or_fake(pred[0])}")
    else:
        st.write(f"Invalid video file: {video_path}. Please provide a valid video file.")
    return result

def main():
    st.title("Deepfake Detection")
    st.write("Upload a video file or provide a YouTube link to analyze for deepfake detection.")

    model_path = "cvit2_deepfake_detection_ep_50.pth"
    model = load_cvit(model_path, fp16=False)

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mpeg", "mpg"])
    youtube_url = st.text_input("YouTube URL")

    if uploaded_file is not None:
        video_path = os.path.join("uploads", uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write("Video uploaded successfully.")
        result = analyze_video(video_path, model)
        st.write(result)

    if youtube_url:
        st.write("Downloading YouTube video...")
        download_youtube_video(youtube_url)
        video_path = os.path.join("downloads", os.listdir("downloads")[0])
        st.write("YouTube video downloaded successfully.")
        result = analyze_video(video_path, model)
        st.write(result)

    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Deepfake Detection Report", ln=True, align="C")
        pdf.cell(200, 10, txt=f"Model: {model_path}", ln=True, align="L")
        pdf.cell(200, 10, txt="Results:", ln=True, align="L")
        for i, name in enumerate(result["video"]["name"]):
            pdf.cell(200, 10, txt=f"Video: {name}", ln=True, align="L")
            pdf.cell(200, 10, txt=f"Prediction: {result['video']['pred_label'][i]}", ln=True, align="L")
            pdf.cell(200, 10, txt=f"Confidence Score: {result['video']['pred'][i]}", ln=True, align="L")
            pdf.cell(200, 10, txt=f"Class: {result['video']['klass'][i]}", ln=True, align="L")
            pdf.cell(200, 10, txt=f"Correct Label: {result['video']['correct_label'][i]}", ln=True, align="L")
            pdf.cell(200, 10, txt="", ln=True, align="L")
        pdf_output_path = os.path.join("result", "report.pdf")
        pdf.output(pdf_output_path)
        st.write(f"PDF report generated: {pdf_output_path}")

if __name__ == "__main__":
    main()
