from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
from io import BytesIO
import numpy as np
from PIL import Image
import pywt
import moviepy.editor as mp
from pydub import AudioSegment
from scipy.io import wavfile
import base64
import cv2

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Menetapkan direktori unggahan
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('temp', exist_ok=True)

# Set path ffmpeg secara eksplisit
AudioSegment.converter = "C:/ffmpeg/bin/ffmpeg.exe"  # Ganti dengan lokasi ffmpeg yang sesuai

# Fungsi untuk menghitung ukuran file dalam format yang lebih mudah dibaca
def calculate_size(file_path):
    return os.path.getsize(file_path)

def compress_image(image):
    image = np.array(image.convert('L'))
    
    # Terapkan DWT pada gambar
    coeffs2 = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs2
    
    # Kuantisasi (mengatur koefisien frekuensi tinggi ke 0)
    thresh = 50
    cH[np.abs(cH) < thresh] = 0
    cV[np.abs(cV) < thresh] = 0
    cD[np.abs(cD) < thresh] = 0
    
    # Rekonstruksi gambar dari koefisien DWT yang terkompresi
    compressed_image = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    compressed_image = np.clip(compressed_image, 0, 255)
    compressed_image = Image.fromarray(compressed_image.astype(np.uint8))
    
    return compressed_image

def compress_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    # Menggunakan wavelet 'haar' untuk DWT pada setiap frame
    transform_method = 'haar'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = os.path.join('temp', 'compressed_video.avi')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Konversi frame menjadi grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Lakukan DWT pada frame grayscale
        coeffs = pywt.dwt2(gray_frame, transform_method)

        # Ambil koefisien approximation dan detail
        cA, (cH, cV, cD) = coeffs

        # Kuantisasi detail koefisien (setting koefisien yang sangat kecil menjadi 0)
        threshold = 50
        cH[np.abs(cH) < threshold] = 0
        cV[np.abs(cV) < threshold] = 0
        cD[np.abs(cD) < threshold] = 0

        # Rekonstruksi frame dari koefisien DWT yang terkompresi
        compressed_frame = pywt.idwt2((cA, (cH, cV, cD)), transform_method)
        compressed_frame = np.clip(compressed_frame, 0, 255)
        compressed_frame = np.uint8(compressed_frame)

        # Tulis frame yang telah dikompresi ke file output
        out.write(cv2.cvtColor(compressed_frame, cv2.COLOR_GRAY2BGR))

    cap.release()
    out.release()
    
    return output_path

def dwt_compress_audio(file_path):
    try:
        # Baca file audio menggunakan scipy
        sample_rate, samples = wavfile.read(file_path)

        # Ambil panjang sampel yang merupakan kelipatan dari 2 untuk kompresi DWT
        length = len(samples)
        new_length = int(np.floor(length / 2)) * 2  # Pastikan panjang sampel genap
        samples = samples[:new_length]

        # Terapkan DWT pada sampel audio
        coeffs = pywt.wavedec(samples, 'db4', level=3)

        # Hapus detail yang kurang signifikan (coef cD3 dan cD2)
        coeffs = list(coeffs)
        coeffs[1:] = (pywt.threshold(cD, np.std(cD)/2, mode="soft") for cD in coeffs[1:])

        # Kembalikan audio yang dikompresi dengan IDWT
        compressed_signal = pywt.waverec(coeffs, 'db4')

        # Simpan file yang dikompresi
        compressed_file_path = os.path.join('temp', 'compressed_audio.wav')
        wavfile.write(compressed_file_path, sample_rate, compressed_signal.astype(np.int16))

        return compressed_file_path

    except Exception as e:
        print(f"Error in dwt_compress_audio: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files['file']
    original_size = len(file.read())
    file.seek(0)
    
    image = Image.open(file.stream)
    compressed_image = compress_image(image)
    
    buf = BytesIO()
    compressed_image.save(buf, format='JPEG')
    buf.seek(0)
    
    compressed_data = buf.getvalue()
    compressed_size = len(compressed_data)
    
    return jsonify({
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compressed_image": compressed_data.decode('latin1')
    })

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['file']
    file_path = os.path.join("temp", file.filename)
    file.save(file_path)
    
    original_size = os.path.getsize(file_path)
    
    compressed_video_path = compress_video(file_path)
    
    compressed_size = os.path.getsize(compressed_video_path)
    
    return jsonify({
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compressed_path": compressed_video_path
    })

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    try:
        # Simpan file audio yang diunggah sementara
        uploaded_file = request.files['file']
        uploaded_file.save('temp.wav')

        # Kompresi file audio menggunakan DWT
        compressed_file_path = dwt_compress_audio('temp.wav')

        if compressed_file_path:
            # Hitung ukuran file asli dan file yang dikompresi
            original_size = calculate_size('temp.wav')
            compressed_size = calculate_size(compressed_file_path)

            # Konversi file yang dikompresi ke base64 untuk dikirimkan ke klien
            with open(compressed_file_path, 'rb') as file:
                compressed_audio_data = file.read()
                compressed_audio_base64 = base64.b64encode(compressed_audio_data).decode('utf-8')

            # Hapus file sementara
            os.remove('temp.wav')
            os.remove(compressed_file_path)

            # Kirim respons ke klien dengan ukuran file dan data audio yang dikompresi
            return jsonify({
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compressed_audio': compressed_audio_base64
            })

        else:
            return jsonify({'error': 'Failed to compress audio'}), 500

    except Exception as e:
        print(f"Error in upload_audio: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Endpoint untuk mendownload file audio yang dikompresi
@app.route('/download/audio', methods=['POST'])
def download_audio():
    try:
        data = request.get_json()

        if 'compressed_audio' in data:
            # Dekode base64 dan simpan ke file sementara
            compressed_audio_data = base64.b64decode(data['compressed_audio'])
            with open('compressed_audio.wav', 'wb') as file:
                file.write(compressed_audio_data)

            # Kirim file yang diunduh ke klien
            return send_file('compressed_audio.wav', as_attachment=True)

        else:
            return jsonify({'error': 'Invalid request'}), 400

    except Exception as e:
        print(f"Error in download_audio: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/download/<file_type>', methods=['POST'])
def download(file_type):
    if file_type == 'image':
        file_data = request.json['compressed_image'].encode('latin1')
        buf = BytesIO(file_data)
        return send_file(buf, mimetype='image/jpeg')
    elif file_type == 'video':
        file_path = request.json['compressed_path']
        return send_file(file_path, mimetype='video/mp4')
    elif file_type == 'audio':
        file_path = request.json['compressed_path']
        return send_file(file_path, mimetype='audio/mpeg')
    else:
        return "Invalid file type", 400

if __name__ == '__main__':
    if not os.path.exists("temp"):
        os.makedirs("temp")
    app.run(debug=True)
