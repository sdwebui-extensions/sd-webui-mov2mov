// This value stores the 'setInterval' used to call the 'updateVideoInfo' function
let mov2movVideoHelperInfoUpdater;

function updateVideoInfo() {
  try {
    // Get the video element
    const inputVideo = gradioApp().querySelector('#mov2mov_mov video');
    if (!inputVideo) {
      // Clear UI
      gradioApp().querySelector('#mov2mov_input_video_info_width textarea').value = 'N/A';
      gradioApp().querySelector('#mov2mov_input_video_info_height textarea').value = 'N/A';
      // Stop updating if video element is not present anymore
      clearInterval(mov2movVideoHelperInfoUpdater);
    } else if (inputVideo.readyState >= 1) {
      // Update UI with video width and height
      gradioApp().querySelector('#mov2mov_input_video_info_width textarea').value = inputVideo.videoWidth;
      gradioApp().querySelector('#mov2mov_input_video_info_height textarea').value = inputVideo.videoHeight;
      // Stop updating once video metadata is loaded
      clearInterval(mov2movVideoHelperInfoUpdater);
    }
  } catch {}
}

// This function is called when a video is loaded or unloaded in the Gradio UI (under the tab "Input video" of mov2mov)
function refreshMov2movVideoHelper() {
  clearInterval(mov2movVideoHelperInfoUpdater);
  mov2movVideoHelperInfoUpdater = setInterval(updateVideoInfo, 100);
}
