let mov2movVideoHelper;

function updateMov2movVideoHelper() {
  try {
    const inputVideo = gradioApp().querySelector('#mov2mov_mov video');
    if (!inputVideo) {
      //gradioApp().querySelector('#mov2mov_input_video_info').style.display = 'none';
      gradioApp().querySelector('#mov2mov_input_video_info_width textarea').value = 'N/A';
      gradioApp().querySelector('#mov2mov_input_video_info_height textarea').value = 'N/A';
    } else {
      //gradioApp().querySelector('#mov2mov_input_video_info').style.display = 'block';
      gradioApp().querySelector('#mov2mov_input_video_info_width textarea').value = inputVideo.videoWidth;
      gradioApp().querySelector('#mov2mov_input_video_info_height textarea').value = inputVideo.videoHeight;
    }
  } catch {}
}

function startMov2movVideoHelper() {
  if (!mov2movVideoHelper) mov2movVideoHelper = setInterval(updateMov2movVideoHelper, 500);
}
