import datetime
import time

import jax
from jax import numpy as jnp
import numpy as np
import os
import signal
import sys
import random

from PySide6 import QtCore, QtWidgets, QtGui

import config_block
import video_processor

signal.signal(signal.SIGINT, signal.SIG_DFL)

# Allowed extensions for videos.
_ALLOWED_EXTENSIONS = [ 'mp4', 'mkv', 'mov', 'avi' ]

def _pretty_duration(seconds: float, total_seconds: float | None = None) -> str:
  to_convert = datetime.timedelta(seconds=seconds)
  # If provided, use total_seconds to determine format (whether to show hour or not), and use it to format seconds.
  if total_seconds is None:
    formatting_duration = to_convert
  else:
    formatting_duration = datetime.timedelta(seconds=total_seconds)
  show_hours = formatting_duration >= datetime.timedelta(hours=1)
  if show_hours:
    hours, remainder = divmod(to_convert, datetime.timedelta(hours=1))
    minutes, remainder = divmod(remainder, datetime.timedelta(minutes=1))
    seconds, remainder = divmod(remainder, datetime.timedelta(seconds=1))
    milliseconds, _ = divmod(remainder, datetime.timedelta(milliseconds=1))
    return f'{hours}:{minutes:>02}:{seconds:>02}.{milliseconds:>03}'
  else:
    minutes, remainder = divmod(to_convert, datetime.timedelta(minutes=1))
    seconds, remainder = divmod(remainder, datetime.timedelta(seconds=1))
    milliseconds, _ = divmod(remainder, datetime.timedelta(milliseconds=1))
    assert minutes <= 59
    return f'{minutes:>02}:{seconds:>02}.{milliseconds:>03}'

def _monospace_font() -> QtGui.QFont:
  # It looks like different platforms require different style hints:
  # https://stackoverflow.com/questions/18896933/qt-qfont-selection-of-a-monospace-font-doesnt-work
  f = QtGui.QFont('monospace')
  f.setStyleHint(QtGui.QFont.Monospace)
  if QtGui.QFontInfo(f).fixedPitch():
    return f
  f.setStyleHint(QtGui.QFont.TypeWriter)
  if QtGui.QFontInfo(f).fixedPitch():
    return f
  f.setFamily('courier')
  return f

class MainWidget(QtWidgets.QWidget):

  selected_video_changed = QtCore.Signal(str)
  request_one_frame = QtCore.Signal()
  preview_width_height_changed = QtCore.Signal(int, int)
  unload_video = QtCore.Signal()
  seek_requested = QtCore.Signal(float)

  def __init__(self):
    super().__init__()

    self.setWindowTitle('ReefShader')

    self._opened_files = []
    self._common_prefix = ''
    self._video_info = None
    self._current_video_file = None

    self._video_processor_thread = QtCore.QThread()
    self._video_processor = video_processor.VideoProcessor()
    self._video_processor.moveToThread(self._video_processor_thread)

    # This keeps track of whether we have a frame request pending. If we do, there's no point queuing up seek signals,
    # because by the time the frame returns, we may want to be somewhere else already. This is mostly for dragging
    # the timeline slider, which would otherwise generate a lot of seek signals, and a lot of unnecessary work for the
    # video processor. Instead, we just store where we want to seek to when the frame request comes back, and that can
    # be updated multiple times while a frame is pending.
    self._frame_request_pending = False
    self._next_seek_to_time = None

    self._is_playing = False
    self._next_frame_display_time = 0.0

    # Left file list + file info.
    self._path_prefix_label = QtWidgets.QLabel('./')
    self._path_prefix_label.setWordWrap(True)
    self._file_list = QtWidgets.QListWidget()
    self._file_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
    add_files_button = QtWidgets.QPushButton('Add File(s)')
    add_folder_button = QtWidgets.QPushButton('Add Folder')
    self._remove_file_button = QtWidgets.QPushButton('Remove')
    self._remove_file_button.setEnabled(False)

    add_files_button.clicked.connect(self.open_files_dialog)
    add_folder_button.clicked.connect(self.open_dir_dialog)
    self._remove_file_button.clicked.connect(self.remove_file_clicked)
    self._file_list.itemSelectionChanged.connect(self.video_multi_selection_changed)
    self._file_list.currentTextChanged.connect(self.video_single_selection_changed)

    # Middle preview display.
    self._opened_file_label = QtWidgets.QLabel()
    self._preview_pixmap = QtGui.QPixmap(QtCore.QSize(800, 600))
    self._preview_pixmap.fill(QtGui.QColor('darkgray'))
    self._preview_frame_label = QtWidgets.QLabel()
    self._preview_frame_label.setPixmap(self._preview_pixmap)
    self._preview_play_stop_button = QtWidgets.QPushButton('⏵')
    self._frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    self._video_position_text = QtWidgets.QLabel('00:00')
    self._video_position_text.setFont(_monospace_font())
    self._preview_enable_checkbox = QtWidgets.QCheckBox('Preview')
    self._preview_enable_checkbox.setChecked(True)

    # Processing status and output folder.
    output_path_label = QtWidgets.QLabel('Output path (relative to file): ')
    self._output_path_field = QtWidgets.QLineEdit('processed/')
    self._process_button = QtWidgets.QPushButton('Process Selected')
    self._process_progress_bar = QtWidgets.QProgressBar()
    self._process_progress_text = QtWidgets.QLabel(f'30% 15 FPS (video 3/15)')

    # Left panel (input files and media info).
    file_list_controls_layout = QtWidgets.QHBoxLayout()
    file_list_controls_layout.addWidget(add_files_button)
    file_list_controls_layout.addWidget(add_folder_button)
    file_list_controls_layout.addWidget(self._remove_file_button)
    input_files_group = QtWidgets.QGroupBox('Input Files')
    input_files_group_v_layout = QtWidgets.QVBoxLayout(input_files_group)
    input_files_group_v_layout.addWidget(self._path_prefix_label)
    input_files_group_v_layout.addWidget(self._file_list)
    input_files_group_v_layout.addLayout(file_list_controls_layout)
    left_v_layout = QtWidgets.QVBoxLayout()
    left_v_layout.addWidget(input_files_group)
    media_info_group = QtWidgets.QGroupBox('Media Info')
    media_info_group_layout = QtWidgets.QVBoxLayout(media_info_group)
    self._media_info = QtWidgets.QLabel('')
    media_info_group_layout.addWidget(self._media_info)
    left_v_layout.addWidget(media_info_group)

    # Middle panel (preview).
    mid_v_layout = QtWidgets.QVBoxLayout()
    mid_v_layout.addWidget(self._opened_file_label)
    mid_v_layout.addWidget(self._preview_frame_label)

    self._preview_controls_container = QtWidgets.QWidget()
    preview_controls_layout = QtWidgets.QHBoxLayout(self._preview_controls_container)
    preview_controls_layout.addWidget(self._preview_play_stop_button, 0)
    preview_controls_layout.addWidget(self._frame_slider, 1)
    preview_controls_layout.addWidget(self._video_position_text, 0)
    preview_controls_layout.addWidget(self._preview_enable_checkbox)
    preview_controls_layout.setSpacing(5)
    preview_controls_layout.setAlignment(QtCore.Qt.AlignTop)
    mid_v_layout.addWidget(self._preview_controls_container)
    mid_v_layout.setAlignment(QtCore.Qt.AlignVCenter)

    # Middle lower panel (output path setting and processing controls).
    output_path_layout = QtWidgets.QHBoxLayout()
    output_path_layout.addWidget(output_path_label)
    output_path_layout.addWidget(self._output_path_field)
    output_path_layout.addWidget(self._process_button)
    output_path_layout.setSpacing(5)
    output_path_layout.setAlignment(QtCore.Qt.AlignTop)

    mid_v_layout.addLayout(output_path_layout, 0)

    mid_v_layout.addWidget(self._process_progress_bar, 0)
    mid_v_layout.addWidget(self._process_progress_text, 0)

    # Option panels.
    option_blocks_v_layout = QtWidgets.QVBoxLayout()

    resolution_block_spec = config_block.ConfigBlockSpec(
      block_name='Resolution Scaling',
      checkable=True,
      elements=[
        config_block.ConfigEnum(key='width', display_name='Width', default_index=0, options=[('1920', 1920), ('1280', 1280)]),
        config_block.ConfigBlockDescription(key='', display_name='', text='Height will be automatically set to preserve aspect ratio.')
      ]
    )

    self._resolution_block = config_block.ConfigBlock(config_block_spec=resolution_block_spec)
    option_blocks_v_layout.addWidget(self._resolution_block)

    gamma_block_spec = config_block.ConfigBlockSpec(
      block_name='Gamma (Contrast) Correction',
      checkable=True,
      elements=[
        config_block.ConfigFloat(key='gamma', display_name='Gamma Correction', default_value=1.0, min_value=0.5, max_value=2.0, resolution=0.01, places=2),
      ]
    )

    self._gamma_block = config_block.ConfigBlock(config_block_spec=gamma_block_spec)
    option_blocks_v_layout.addWidget(self._gamma_block)

    normalisation_block_spec = config_block.ConfigBlockSpec(
      block_name='Colour Normalisation',
      checkable=True,
      elements=[
        config_block.ConfigFloat(key='max_gain', display_name='Max Gain', default_value=10, min_value=1, max_value=25, places=1),
        config_block.ConfigFloat(key='temporal_smoothing', display_name='Temporal Smoothing', default_value=0.95, min_value=0.0, max_value=1.0, resolution=0.001, places=3),
      ]
    )

    self._normalisation_block = config_block.ConfigBlock(config_block_spec=normalisation_block_spec)
    option_blocks_v_layout.addWidget(self._normalisation_block)

    encode_block_spec = config_block.ConfigBlockSpec(
      block_name='Video Encode',
      checkable=False,
      elements=[
        config_block.ConfigEnum(key='encoder', display_name='Encoder', default_index=0, options=[
            ('H264 (8-bit)', 'h264'),
            ('HEVC (8-bit)', 'hevc'),
            ('HEVC (10-bit)', 'hevc10'),
        ]),
        config_block.ConfigInt(key='bitrate', display_name='Bit Rate (mbps)', default_value=20, min_value=1, max_value=200),
      ]
    )

    self._encode_block = config_block.ConfigBlock(config_block_spec=encode_block_spec)
    option_blocks_v_layout.addWidget(self._encode_block)

    option_blocks_v_layout.addStretch(1)

    root_h_layout = QtWidgets.QHBoxLayout(self)
    root_h_layout.addLayout(left_v_layout)
    root_h_layout.addLayout(mid_v_layout)
    root_h_layout.addLayout(option_blocks_v_layout)

    # Connections
    self.selected_video_changed.connect(self._video_processor.request_load_video)
    self.request_one_frame.connect(self._video_processor.request_one_frame)
    self.seek_requested.connect(self._video_processor.request_seek_to)
    self.preview_width_height_changed.connect(self._video_processor.set_preview_width_height)
    self.preview_width_height_changed.emit(800, 600)
    self.unload_video.connect(self._video_processor.unload_video)
    self._video_processor.frame_decoded.connect(self.frame_received)
    self._video_processor.new_video_info.connect(self.update_video_info)
    self._frame_slider.sliderMoved.connect(self.frame_slider_moved)
    self._frame_slider.sliderPressed.connect(self.frame_slider_pressed)
    self._preview_play_stop_button.clicked.connect(self._play_stop_clicked)

    self._video_processor_thread.start()

    self.video_multi_selection_changed()

  @QtCore.Slot()
  def open_files_dialog(self):
    file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open one or more files', '', f'Videos ({" ".join(['*.' + ext for ext in _ALLOWED_EXTENSIONS])})')
    if file_names:
      self.add_files_impl(file_names)

  @QtCore.Slot()
  def open_dir_dialog(self):
    dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Add all files in directory', '')
    files_found = []
    if dir_path:
      for filename in os.listdir(dir_path):
        if filename[-3:].lower() in _ALLOWED_EXTENSIONS:
          files_found.append(os.path.join(dir_path, filename))
    if files_found:
      self.add_files_impl(files_found)

  @QtCore.Slot()
  def remove_file_clicked(self):
    for item in self._file_list.selectedItems():
      full_path = os.path.join(self._common_prefix, item.text())
      assert full_path in self._opened_files
      self._opened_files.remove(full_path)
    self.opened_files_updated()

  @QtCore.Slot()
  def video_single_selection_changed(self, filename):
    # This function deals with loading new video for preview, so it's called each time with the last
    # selected video. If the user selects over multiple videos with her mouse, this gets called every
    # time a video is added to the selection set.
    if filename:
      full_path = os.path.join(self._common_prefix, filename)
      self._opened_file_label.setText(full_path)
      self._set_playing(False)
      print(f'sending {full_path}')
      self.selected_video_changed.emit(full_path)
      self._request_new_frame()
      self._current_video_file = full_path

  @QtCore.Slot()
  def video_multi_selection_changed(self):
    # This function deals with changes that aren't dealt with by video_single_selection_changed.
    selected_items = self._file_list.selectedItems()
    if not selected_items:
      self._remove_file_button.setEnabled(False)
      self._process_button.setEnabled(False)
      self.unload_video.emit()
      self._disable_preview()
      self._media_info.setText('')
      self._set_playing(False)
    else:
      self._remove_file_button.setEnabled(True)
      self._process_button.setEnabled(True)

  @QtCore.Slot()
  def update_video_info(self, video_info):
    self._video_info = video_info
    self._media_info.setText(
      f'Resolution: {video_info.width}x{video_info.height}\n'
      f'Frame rate: {video_info.frame_rate:.2f}\n'
      f'Duration: {_pretty_duration(video_info.duration)}\n'
      f'Num Frames: {video_info.num_frames}\n'
      f'Decoder: {video_info.decoder_name}')
    self._frame_slider.setValue(0)
    self._frame_slider.setMinimum(0)
    self._frame_slider.setMaximum(video_info.num_frames)
    self._preview_controls_container.setEnabled(True)

  @QtCore.Slot()
  def frame_received(self, frame_data: jnp.ndarray | None, frame_time: float | None):
    if frame_data is not None:
      now = time.time()
      if now >= self._next_frame_display_time:
        self._update_preview(frame_data, frame_time)
      else:
        delay_ms = round((self._next_frame_display_time - now) * 1000)
        QtCore.QTimer.singleShot(delay_ms, lambda: self._update_preview(frame_data, frame_time, now + delay_ms / 1000))
    else:
      # frame_data can be None if we request a frame but there is no frame left.
      self._video_position_text.setText(_pretty_duration(self._video_info.duration, self._video_info.duration))
      self._frame_slider.setValue(self._frame_slider.maximum())
      self._set_playing(False)

    self._frame_request_pending = False
    if self._next_seek_to_time is not None:
      self.seek_requested.emit(self._next_seek_to_time)
      self._request_new_frame()
      self._next_seek_to_time = None

  @QtCore.Slot()
  def _update_preview(self, frame_data: jnp.ndarray | None, frame_time: float | None, exp = 0) -> None:
    now = time.time()
    height, width = frame_data.shape[:2]
    qimage = QtGui.QImage(frame_data, width, height, 3 * width, QtGui.QImage.Format_RGB888)
    self._preview_pixmap = QtGui.QPixmap(QtGui.QPixmap.fromImage(qimage))
    self._preview_frame_label.setPixmap(self._preview_pixmap)
    self._preview_frame_label.setScaledContents(True)
    self._preview_frame_label.update()

    # Use duration to format frame time, so that if the video is over an hour, frame time is always shown
    # with the hour field.
    self._video_position_text.setText(_pretty_duration(frame_time, self._video_info.duration))

    if self._is_playing:
      frame_duration = 1.0 / self._video_info.frame_rate
      slider_value = round((frame_time / self._video_info.duration) * self._video_info.num_frames)
      self._frame_slider.setValue(slider_value)
      self._next_frame_display_time = now + frame_duration
      self._request_new_frame()

  def _request_new_frame(self):
    self._frame_request_pending = True
    self.request_one_frame.emit()

  def _schedule_seek(self, frame_time, stop_playing=True):
    if self._frame_request_pending:
      self._next_seek_to_time = frame_time
    else:
      self._next_seek_to_time = None
      self.seek_requested.emit(frame_time)
      self._request_new_frame()
    if stop_playing:
      self._set_playing(False)

  @QtCore.Slot()
  def frame_slider_moved(self, new_value):
    frame_time = new_value * 1.0 / self._video_info.frame_rate
    self._schedule_seek(frame_time)

  @QtCore.Slot()
  def frame_slider_pressed(self):
    frame_time = self._frame_slider.value() * 1.0 / self._video_info.frame_rate
    self._schedule_seek(frame_time)

  @QtCore.Slot()
  def _play_stop_clicked(self):
    self._set_playing(not self._is_playing)

  def _set_playing(self, playing):
    self._is_playing = playing
    self._preview_play_stop_button.setText('⏹' if playing else '⏵')
    if playing:
      if self._frame_slider.value() == self._frame_slider.maximum():
        self._schedule_seek(0.0, stop_playing=False)  # This triggers a request new frame when seek happens.
        self._frame_slider.setValue(0)
      else:
        self._request_new_frame()

  def _disable_preview(self):
    self._preview_pixmap = QtGui.QPixmap(QtCore.QSize(800, 600))
    self._preview_pixmap.fill(QtGui.QColor('darkgray'))
    self._preview_frame_label.setPixmap(self._preview_pixmap)
    self._preview_frame_label.update()
    self._frame_slider.setValue(0)
    self._preview_controls_container.setEnabled(False)

  def add_files_impl(self, file_names):
    for file_name in file_names:
      if file_name not in self._opened_files:
        self._opened_files.append(file_name)
    self.opened_files_updated()

  def opened_files_updated(self):
    self._file_list.clear()
    if len(self._opened_files) == 0:
      self._remove_file_button.setEnabled(False)
    else:
      if len(self._opened_files) == 1:
        # Special case - use the parent directory as the common path.
        self._common_prefix = os.path.dirname(self._opened_files[0])
      else:
        self._common_prefix = os.path.commonpath(self._opened_files)
      self._path_prefix_label.setText(self._common_prefix)
      for file_name in self._opened_files:
        short_name = file_name[(len(self._common_prefix) + 1):]
        assert os.path.join(self._common_prefix, short_name) == file_name
        QtWidgets.QListWidgetItem(short_name, self._file_list)
    

if __name__ == "__main__":
  app = QtWidgets.QApplication([])

  widget = MainWidget()
  widget.show()

  sys.exit(app.exec())