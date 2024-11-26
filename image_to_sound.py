import os
import sys  # Import sys for path adjustments
import time
import logging
import threading
import signal

from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from pydub import AudioSegment
import numpy as np
from scipy.signal import butter, lfilter
from pydub.utils import which

# Explicitly set FFmpeg path
AudioSegment.converter = which("ffmpeg")
if not AudioSegment.converter:
    raise RuntimeError("FFmpeg not found! Ensure it is installed and added to PATH.")

# Determine base path for templates and static files
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle (compiled with PyInstaller)
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

# Initialize Flask app with adjusted paths
app = Flask(__name__, template_folder=os.path.join(base_path, 'templates'), static_folder=os.path.join(base_path, 'static'))
app.secret_key = 'your_secure_random_secret_key'  # Replace with a secure random key in production

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure upload and output folders
UPLOAD_FOLDER = os.path.join(base_path, 'uploads')
OUTPUT_FOLDER = os.path.join(base_path, 'output')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB limit

def create_output_subfolder():
    """Create an output subfolder with a timestamp."""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_subfolder = os.path.join(OUTPUT_FOLDER, f"output_{timestamp}")
    os.makedirs(output_subfolder, exist_ok=True)
    logging.debug(f"Created output subfolder: {output_subfolder}")
    return output_subfolder

def butter_lowpass(cutoff, fs, order=5):
    """Design a Butterworth low-pass filter."""
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    """Apply a low-pass filter to the data."""
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def moving_average(x, N):
    """Compute the moving average of the data."""
    return np.convolve(x, np.ones(N)/N, mode='same')

def image_to_sound(image_path, frame_rate, read_direction="horizontal", speed_factor=2):
    """
    Convert an image to audio samples with improved quality and lower pitch.

    Args:
        image_path (str): Path to the image file.
        frame_rate (int): Frames per second.
        read_direction (str): Direction to read pixels ("horizontal" or "vertical").
        speed_factor (float): Factor to lower playback speed (increase pitch lowering).

    Returns:
        np.ndarray: Array of int16 audio samples.
    """
    logging.debug(f"Processing image: {image_path}")
    try:
        img = Image.open(image_path).convert("L")  # Convert to grayscale
        pixels = np.array(img)

        if read_direction == "horizontal":
            pixels = pixels.flatten()
        elif read_direction == "vertical":
            pixels = pixels.T.flatten()  # Transpose for top-to-bottom reading

        # Adjust the sample rate to lower the pitch
        sample_rate = int(44100 / speed_factor)  # Lower the sample rate
        samples_per_frame = int(sample_rate / frame_rate)

        if len(pixels) < samples_per_frame:
            pixels = np.pad(pixels, (0, samples_per_frame - len(pixels)), mode='constant')

        indices = np.linspace(0, len(pixels) - 1, samples_per_frame).astype(int)
        resampled_pixels = pixels[indices]

        # Apply moving average to smooth the pixel data
        N = 5  # Window size for moving average
        resampled_pixels = moving_average(resampled_pixels, N)

        # Scale pixel values to float in range -1.0 to 1.0
        samples = np.interp(resampled_pixels, (0, 255), (-1.0, 1.0))

        # Apply low-pass filter to samples
        cutoff = 1000  # Hz, adjust as needed
        order = 6      # Filter order
        samples = lowpass_filter(samples, cutoff, sample_rate, order)

        # Scale back to int16 range
        samples = np.int16(samples * 32767)

        return samples
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None

def batch_process_images(file_paths, output_folder, frame_rate, export_type, read_direction, progress_dict, task_id, speed_factor=2):
    """
    Process a batch of images to generate audio.

    Args:
        file_paths (list): List of image file paths.
        output_folder (str): Directory to save output files.
        frame_rate (int): Frames per second.
        export_type (str): Type of export ("audio").
        read_direction (str): Direction to read pixels ("horizontal" or "vertical").
        progress_dict (dict): Dictionary to track progress.
        task_id (str): Unique identifier for the task.
        speed_factor (float): Factor to lower playback speed (increase pitch lowering).
    """
    logging.info("Starting batch processing of images.")
    images = sorted(file_paths, key=lambda x: os.path.basename(x))
    logging.debug(f"Sorted images: {images}")

    all_samples = []
    total_images = len(images)

    for index, image_path in enumerate(images, start=1):
        samples = image_to_sound(image_path, frame_rate, read_direction, speed_factor)
        if samples is not None:
            all_samples.append(samples)
            logging.debug(f"Processed audio samples for image {index}: {image_path}")
        progress_dict[task_id] = int((index / total_images) * 50)  # Processing images (0-50%)

    if export_type in ("audio",):
        if not all_samples:
            logging.warning("No audio samples to concatenate.")
            progress_dict[task_id] = 100
            return
        else:
            try:
                # Concatenate all samples into one array
                concatenated_samples = np.concatenate(all_samples)
                logging.debug(f"Total concatenated samples: {len(concatenated_samples)}")

                # Create AudioSegment from concatenated samples
                full_audio = AudioSegment(
                    concatenated_samples.tobytes(),
                    frame_rate=int(44100 / speed_factor),  # Adjust sample rate for lower pitch
                    sample_width=2,  # 16 bits
                    channels=1
                )

                # Export concatenated audio as MP3 with high bitrate
                audio_output = os.path.join(output_folder, "output_audio.mp3")
                full_audio.export(audio_output, format="mp3", bitrate="320k")
                logging.debug(f"Saved concatenated audio file: {audio_output}")

                # Update progress
                progress_dict[task_id] = 100  # Processing complete
            except Exception as e:
                logging.error(f"Error exporting concatenated audio: {e}")
                progress_dict[task_id] = -1  # Indicate failure

def clear_files():
    """Delete all files in the uploads and output directories."""
    folders = [UPLOAD_FOLDER, OUTPUT_FOLDER]
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    logging.debug(f"Deleted file: {file_path}")
            except Exception as e:
                logging.error(f"Error deleting file {file_path}: {e}")

# Dictionary to track progress of tasks
progress = {}

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route('/favicon.ico')
def favicon():
    return '', 204  # Silences missing favicon error

@app.route('/output/<path:filename>', methods=['GET'])
def serve_output_file(filename):
    """Serve files from the output folder."""
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route("/convert", methods=["POST"])
def convert():
    if 'files' not in request.files:
        flash('No file part')
        logging.warning("No file part in the request.")
        return redirect(request.url)
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        flash('No selected files')
        logging.warning("No files selected for upload.")
        return redirect(request.url)

    # Save uploaded files
    saved_file_paths = []
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
            saved_file_paths.append(file_path)
            logging.debug(f"Saved uploaded file: {file_path}")
        except Exception as e:
            logging.error(f"Error saving file {filename}: {e}")

    if not saved_file_paths:
        flash('No valid files uploaded.')
        logging.warning("No valid files were uploaded.")
        return redirect(request.url)

    # Get form data
    try:
        frame_rate = int(request.form.get('frame_rate', 24))
        read_direction = request.form.get('read_direction', 'horizontal')
        export_type = request.form.get('export_type', 'audio')
        speed_factor = float(request.form.get('speed_factor', 2))  # Added speed factor
        logging.debug(f"Received form data - Frame Rate: {frame_rate}, Read Direction: {read_direction}, Export Type: {export_type}, Speed Factor: {speed_factor}")
    except ValueError as e:
        flash('Invalid input values.')
        logging.error(f"Invalid input values: {e}")
        return redirect(request.url)

    # Create output subfolder
    output_subfolder = create_output_subfolder()

    # Generate a unique task ID
    task_id = str(time.time())
    progress[task_id] = 0  # Initialize progress

    # Start processing in a separate thread to prevent blocking
    try:
        thread = threading.Thread(target=batch_process_images, args=(
            saved_file_paths,
            output_subfolder,
            frame_rate,
            export_type,
            read_direction,
            progress,
            task_id,
            speed_factor
        ))
        thread.start()
        logging.info("Started batch processing thread.")
    except Exception as e:
        flash('Failed to start processing.')
        logging.error(f"Failed to start processing thread: {e}")
        return redirect(request.url)

    return jsonify({"task_id": task_id, "output_folder": output_subfolder})

@app.route("/progress/<task_id>", methods=["GET"])
def get_progress(task_id):
    """Endpoint to get the progress of a task."""
    status = progress.get(task_id, None)
    if status is None:
        return jsonify({"status": "invalid"}), 404
    elif status == -1:
        return jsonify({"status": "error"}), 500
    else:
        return jsonify({"status": "processing", "progress": status})

@app.route("/clear_files", methods=["POST"])
def clear_uploaded_files():
    """Endpoint to clear uploaded and output files."""
    try:
        clear_files()
        flash('All uploaded and output files have been cleared.')
        logging.info("Cleared all uploaded and output files.")
    except Exception as e:
        flash('Failed to clear files.')
        logging.error(f"Failed to clear files: {e}")
    return redirect(url_for('home'))

@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Shut down the Flask server."""
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        os.kill(os.getpid(), signal.SIGTERM)  # Fallback if Werkzeug is not used
    else:
        func()
    return "Server shutting down..."

if __name__ == "__main__":
    # Start the Flask app
    app.run(debug=False, host="127.0.0.1", port=5000)
