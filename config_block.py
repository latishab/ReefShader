from dataclasses import dataclass

from typing import Sequence

from PySide6 import QtCore, QtWidgets, QtGui

# This is basically a dictionary wrapper that keeps track of which fields have been used, to help debug.
class ConfigDict(dict):
  def __init__(self):
    super().__init__()
    self._used_fields = set()

  def __getitem__(self, key):
    self._used_fields.add(key)
    return super().__getitem__(key)

  def unused_fields(self) -> set[str]:
    return set(super().keys()) - self._used_fields

  def unused_fields_recursive(self, prefix='') -> list[str]:
    ret = []
    unused = self.unused_fields()
    for key in self.keys():
      val = super().__getitem__(key)
      if isinstance(val, ConfigDict):
        ret.extend(val.unused_fields_recursive(prefix=f'{key}->'))
      elif key in unused:
        ret.append(f'{prefix}{key}')
    return ret

  def reset_usage_tracker(self):
    self._used_fields = set()

@dataclass
class ConfigBlockElement:
  key: str
  display_name: str

  def __init__(self, key, display_name):
    super().__init__()
    self.key = key
    self.display_name = display_name

  def value(self):
    raise NotImplementedError()

class ConfigBool(ConfigBlockElement, QtWidgets.QCheckBox):
  updated = QtCore.Signal()

  def __init__(self, key: str, display_name: str, default_value: bool):
    super().__init__(key=key, display_name=display_name)
    self.setText(display_name)
    self.setChecked(default_value)
    self.checkStateChanged.connect(self.updated)

  def value(self):
    return self.isChecked()

class ConfigFloat(ConfigBlockElement, QtWidgets.QVBoxLayout):
  updated = QtCore.Signal()

  def __init__(self, key: str, display_name: str, default_value: float, min_value: float, max_value: float, resolution: int = 0.1, places: int = 2):
    super().__init__(key=key, display_name=display_name)
    self._min_value = min_value
    self._value = default_value
    self._steps = round((max_value - min_value) / resolution)
    self._places = places
    name_and_val_h_layout = QtWidgets.QHBoxLayout()
    name_and_val_h_layout.addWidget(QtWidgets.QLabel(display_name))
    name_and_val_h_layout.addStretch(1)
    self._value_label = QtWidgets.QLabel('0.0')
    name_and_val_h_layout.addWidget(self._value_label)
    self.addLayout(name_and_val_h_layout)
    self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    self._slider.setMinimum(0)
    self._slider.setMaximum(self._steps)
    self._value_range = max_value - min_value
    default_value_steps = round((default_value - min_value) / self._value_range * self._steps)
    self._slider.setValue(default_value_steps)

    self._slider.valueChanged.connect(self.update_value)
    self.update_value()
    self.addWidget(self._slider)

  @QtCore.Slot()
  def update_value(self):
    new_value = self._slider.value() / self._steps * self._value_range + self._min_value
    self._value_label.setText(f'{new_value:.{self._places}f}')
    self._value = new_value
    self.updated.emit()

  def value(self):
    return self._value

class ConfigInt(ConfigBlockElement, QtWidgets.QVBoxLayout):
  updated = QtCore.Signal()

  def __init__(self, key: str, display_name: str, default_value: int, min_value: int, max_value: int):
    super().__init__(key=key, display_name=display_name)
    name_and_val_h_layout = QtWidgets.QHBoxLayout()
    name_and_val_h_layout.addWidget(QtWidgets.QLabel(display_name))
    name_and_val_h_layout.addStretch(1)
    self._value_label = QtWidgets.QLabel('0')
    name_and_val_h_layout.addWidget(self._value_label)
    self.addLayout(name_and_val_h_layout)
    self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    self._slider.setMinimum(min_value)
    self._slider.setMaximum(max_value)
    self._slider.setValue(default_value)

    self._slider.valueChanged.connect(self.update_value)
    self.update_value()
    self.addWidget(self._slider)

  @QtCore.Slot()
  def update_value(self):
    new_value = self._slider.value()
    self._value_label.setText(f'{new_value}')
    self.updated.emit()

  def value(self):
    return self._slider.value()

class ConfigEnum(ConfigBlockElement, QtWidgets.QHBoxLayout):
  updated = QtCore.Signal()

  def __init__(self, key: str, display_name: str, default_index: int, options: list[tuple[str, str]]):
    super().__init__(key=key, display_name=display_name)
    assert default_index >= 0 and default_index < len(options)
    self.addWidget(QtWidgets.QLabel(display_name))
    self._combobox = QtWidgets.QComboBox()
    for enum_option_display, enum_option_val in options:
      self._combobox.addItem(enum_option_display, enum_option_val)
    self._combobox.setCurrentIndex(default_index)
    self.addWidget(self._combobox)
    self._combobox.currentIndexChanged.connect(self.updated)

  def value(self):
    return self._combobox.currentData()

class ConfigPath(ConfigBlockElement, QtWidgets.QVBoxLayout):
  updated = QtCore.Signal()

  def __init__(self, key: str, display_name: str, default_value: str = [], path_filter: str = ''):
    super().__init__(key=key, display_name=display_name)
    self._display_name = display_name
    self._filter = path_filter
    self.addWidget(QtWidgets.QLabel(display_name))
    h_layout = QtWidgets.QHBoxLayout()
    self._line_edit = QtWidgets.QLineEdit()
    open_button = QtWidgets.QPushButton('Open')
    open_button.clicked.connect(self.open_file_dialog)
    h_layout.addWidget(self._line_edit)
    h_layout.addWidget(open_button)
    self.addLayout(h_layout)

  @QtCore.Slot()
  def open_file_dialog(self):
    file_name, _ = QtWidgets.QFileDialog.getOpenFileName(None, self._display_name, '', self._filter)
    if file_name:
      self._line_edit.setText(file_name)
      self.updated.emit()

  def value(self):
    return self._line_edit.text()

@dataclass
class ConfigBlockDescription(ConfigBlockElement):
  text: str

@dataclass
class ConfigBlockSpec:
  block_name: str
  display_name: str
  checkable: bool
  elements: Sequence[ConfigBlockElement]

class ConfigBlock(QtWidgets.QGroupBox):
  updated = QtCore.Signal()

  def __init__(self, config_block_spec: ConfigBlockSpec, parent: QtWidgets.QWidget = None):
    super().__init__(config_block_spec.display_name, parent)
    self.setCheckable(config_block_spec.checkable)

    v_layout = QtWidgets.QVBoxLayout(self)

    self._key = config_block_spec.block_name
    self._elements = config_block_spec.elements

    for element in config_block_spec.elements:
      if isinstance(element, ConfigBool):
        v_layout.addWidget(element)
        element.updated.connect(self.updated)
      elif isinstance(element, ConfigInt):
        v_layout.addLayout(element)
        element.updated.connect(self.updated)
      elif isinstance(element, ConfigFloat):
        v_layout.addLayout(element)
        element.updated.connect(self.updated)
      elif isinstance(element, ConfigEnum):
        v_layout.addLayout(element)
        element.updated.connect(self.updated)
      elif isinstance(element, ConfigPath):
        v_layout.addLayout(element)
        element.updated.connect(self.updated)
      elif isinstance(element, ConfigBlockDescription):
        label = QtWidgets.QLabel(element.text)
        label.setWordWrap(True)
        v_layout.addWidget(label)
      else:
        raise ValueError(f'What do we do with a {option}?')

    self.clicked.connect(self.updated)

  def to_config_dict(self) -> ConfigDict:
    config_dict = ConfigDict()
    if self.isCheckable():
      config_dict['enabled'] = self.isChecked()
    for element in self._elements:
      if isinstance(element, ConfigBlockDescription):
        continue
      else:
        config_dict[element.key] = element.value()
    return config_dict

  def name(self) -> str:
    return self._key

