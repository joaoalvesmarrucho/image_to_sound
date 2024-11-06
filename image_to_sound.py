import os
import time
from flask import Flask, request, render_template, redirect, url_for
from tkinter import Tk
from tkinter.filedialog import askdirectory
from PIL import Image
from pydub import AudioSegment
import numpy as np
from moviepy.editor import ImageSequenceClip, AudioFileClip
import threading
import webbrowser

app = Flask(__name__)
last_folder_path = os.getcwd()  # Default to the program folder

def open_folder_picker():
    """Open a folder picker dialog using tkinter."""
    global last_folder_path
    Tk().withdraw()  # Hide the root tkinter window
    folder_path = askdirectory(initialdir=last_folder_path, title="Select Input Folder")
    if folder_path:
        last_folder_path = folder_path  # Update the last selected folder
    return folder_path

def create_output_folder(input_folder):
    """Create an output folder with a timestamp."""
    timestamp = time.strftime("%Y-%m-%d_%H-%M")
    output_folder = f"{input_folder}_output_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def image_to_sound(image_path, frame_rate, output_folder, index, read_direction="horizontal"):
    img = Image.open(image_path).convert("L")  # Convert to greyscale
    pixels = np.array(img)

    if read_direction == "horizontal":
        pixels = pixels.flatten()
    elif read_direction == "vertical":
        pixels = pixels.T.flatten()  # Transpose for top-to-bottom reading

    samples = np.interp(pixels, (0, 255), (-32768, 32767)).astype(np.int16)
    audio = AudioSegment(
        samples.tobytes(),
        frame_rate=frame_rate,
        sample_width=2,
        channels=1
    )
    output_file = os.path.join(output_folder, f"output_{index}.wav")
    audio.export(output_file, format="wav")
    print(f"Saved: {output_file}")
    return output_file

def images_to_video(images, output_folder, frame_rate, audio_file=None):
    clip = ImageSequenceClip(images, fps=frame_rate)
    if audio_file:
        audio = AudioFileClip(audio_file)
        clip = clip.set_audio(audio)
    video_output = os.path.join(output_folder, "output_video.mp4")
    clip.write_videofile(video_output, codec="libx264")
    print(f"Saved: {video_output}")

def batch_process_images(input_folder, output_folder, frame_rate, export_type, read_direction):
    images = [
        os.path.join(input_folder, filename)
        for filename in sorted(os.listdir(input_folder))
        if filename.endswith((".png", ".jpg", ".jpeg"))
    ]

    audio_files = []
    for index, image_path in enumerate(images, start=1):
        audio_file = image_to_sound(image_path, frame_rate, output_folder, index, read_direction)
        audio_files.append(audio_file)

    if export_type in ("video", "both"):
        images_to_video(images, output_folder, frame_rate, audio_files[0] if export_type == "both" else None)

    print("Processing complete.")

@app.route("/", methods=["GET", "POST"])
def index():
    global last_folder_path
    if request.method == "POST":
        if "select_folder" in request.form:
            # Use tkinter file picker
            input_folder = open_folder_picker()
            if input_folder:
                return redirect(url_for("index", input_folder=input_folder))

        input_folder = request.args.get("input_folder", "")
        frame_rate = int(request.form["frame_rate"])
        read_direction = request.form["read_direction"]
        export_type = request.form["export_type"]

        if input_folder:
            output_folder = create_output_folder(input_folder)
            batch_process_images(input_folder, output_folder, frame_rate, export_type, read_direction)
            return f"Processing complete! Outputs saved in: {output_folder}"

    input_folder = request.args.get("input_folder", last_folder_path)
    return render_template("index.html", input_folder=input_folder)


def open_browser():
    """Open the web browser after the server starts."""
    time.sleep(1)  # Wait briefly to ensure the server is running
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    # Open the Flask app in the default web browser
    webbrowser.open("http://127.0.0.1:5000")
    # Start the Flask app
    app.run(debug=False)