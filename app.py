import cv2
import streamlit as st
import image_dehazer
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import numpy as np

def calculate_metrics(original_frame, dehazed_frame):
    # Convert frames to grayscale
    original_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    dehazed_gray = cv2.cvtColor(dehazed_frame, cv2.COLOR_BGR2GRAY)

    # Calculate PSNR
    psnr = cv2.PSNR(original_gray, dehazed_gray)

    # Calculate SSIM
    _, ssim_value = ssim(original_gray, dehazed_gray, full=True)

    return psnr, ssim_value

def dehaze_video(input_video_path, output_video_path, frame_skip=1, ssim_threshold=0.95):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    # Get the video's frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Create a VideoWriter object to save the processed video with the same dimensions as the input
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 60, (frame_width, frame_height))

    # Create an instance of the image_dehazer
    dehazer = image_dehazer.image_dehazer()

    frame_count = 0
    ssim_values = []

    frame_placeholder = st.empty()
    frame_placeholder_1 = st.empty()
    accuracy_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("The video has ended")
            break

        # Process every "frame_skip" frames and skip the rest
        if frame_count % frame_skip == 0:
            # Process the frame to remove haze
            dehazed_frame, _ = dehazer.remove_haze(frame)

            # Calculate PSNR and SSIM
            psnr, ssim_value = calculate_metrics(frame, dehazed_frame)

            # Write the processed frame to the output video
            out.write(dehazed_frame)

            # Display the original and processed frames
            frame_placeholder.image(frame, channels="RGB")
            frame_placeholder_1.image(dehazed_frame, channels="RGB")

            # Display PSNR and SSIM
            st.text(f"PSNR: {psnr:.2f}")
            st.text(f"SSIM: {ssim_value.mean():.2f}")

            # Store SSIM value for accuracy calculation
            ssim_values.append(ssim_value.mean())

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q') or stop_button_pressed:
            break

    # Release the video objects
    cap.release()
    out.release()

    # Calculate and display mean accuracy
    mean_accuracy = np.mean(ssim_values)
    accuracy_placeholder.text(f"Mean Accuracy: {mean_accuracy:.2f}")

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    st.title("Video Dehazing App")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        input_video_path = Path("input_video.mp4")
        input_video_path.write_bytes(uploaded_file.read())

        # Display the uploaded video
        # st.video(str(input_video_path))

        frame_skip = st.slider("Frame Skip", 1, 50, 20)
        ssim_threshold = st.slider("SSIM Threshold", 0.1, 1.0, 0.95)

        if st.button("Process"):
            output_video_path = Path("output_video.mp4")
            dehaze_video(str(input_video_path), str(output_video_path), frame_skip, ssim_threshold)

            # Display the dehazed video
            st.title("Dehazed Video")
            st.video(str(output_video_path))
