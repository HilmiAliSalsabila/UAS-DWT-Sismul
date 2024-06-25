function formatSize(size) {
  if (size < 1024) {
    return size + ' B';
  } else if (size < 1024 * 1024) {
    return (size / 1024).toFixed(2) + ' KB';
  } else {
    return (size / (1024 * 1024)).toFixed(2) + ' MB';
  }
}

function uploadImage() {
  const input = document.getElementById('imageInput');
  const file = input.files[0];

  const formData = new FormData();
  formData.append('file', file);

  fetch('/upload_image', {
    method: 'POST',
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById('imageSizes').innerText = `Original Size: ${formatSize(data.original_size)}, Compressed Size: ${formatSize(data.compressed_size)}`;

      const img = document.getElementById('compressedImage');
      img.src = 'data:image/jpeg;base64,' + btoa(data.compressed_image);
      img.style.display = 'block';

      document.getElementById('downloadImage').style.display = 'block';
    })
    .catch(console.error);
}

function uploadVideo() {
  const input = document.getElementById('videoInput');
  const file = input.files[0];

  const formData = new FormData();
  formData.append('file', file);

  fetch('/upload_video', {
    method: 'POST',
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById('videoSizes').innerText = `Original Size: ${formatSize(data.original_size)}, Compressed Size: ${formatSize(data.compressed_size)}`;

      const video = document.getElementById('compressedVideo');
      video.src = URL.createObjectURL(new Blob([data.compressed_video], { type: 'video/mp4' }));
      video.style.display = 'block';

      document.getElementById('downloadVideo').style.display = 'block';
    })
    .catch(console.error);
}

function uploadAudio() {
  const input = document.getElementById('audioInput');
  const file = input.files[0];

  const formData = new FormData();
  formData.append('file', file);

  fetch('/upload_audio', {
    method: 'POST',
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById('audioSizes').innerText = `Original Size: ${formatSize(data.original_size)}, Compressed Size: ${formatSize(data.compressed_size)}`;

      const audio = document.getElementById('compressedAudio');
      audio.src = 'data:audio/wav;base64,' + data.compressed_audio; // Ubah tipe audio sesuai format yang dikompresi
      audio.style.display = 'block';

      document.getElementById('downloadAudio').style.display = 'block';
    })
    .catch(console.error);
}

function downloadFile(fileType) {
  let sourceElement, fileExtension;

  if (fileType === 'image') {
    sourceElement = document.getElementById('compressedImage');
    fileExtension = '.jpg';
  } else if (fileType === 'video') {
    sourceElement = document.getElementById('compressedVideo');
    fileExtension = '.mp4';
  } else if (fileType === 'audio') {
    sourceElement = document.getElementById('compressedAudio');
    fileExtension = '.wav'; // Ubah ekstensi berdasarkan format audio yang dikompresi
  }

  if (!sourceElement) {
    console.error('Invalid file type:', fileType);
    return;
  }

  fetch('/download/' + fileType, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(fileType === 'image' ? { compressed_image: sourceElement.src.split(',')[1] } : fileType === 'video' ? { compressed_path: sourceElement.src } : { compressed_path: sourceElement.src }),
  })
    .then((response) => response.blob())
    .then((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = '.none';
      a.href = url;
      a.download = 'compressed_' + fileType + fileExtension;
      document.body.appendChild(a);
      a.click();
      URL.revokeObjectURL(url);
    })
    .catch(console.error);
}
