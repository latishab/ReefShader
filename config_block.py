from dataclasses import dataclass

from typing import Sequence

from PySide6 import QtCore, QtWidgets, QtGui

@dataclass
class ConfigBlockElement:
  key: str
  display_name: str

class ConfigBool(QtWidgets.QCheckBox):
  def __init__(self, key: str, display_name: str, default_value: bool):
    super().__init__(display_name)
    self.setChecked(default_value)

class ConfigFloat(QtWidgets.QVBoxLayout):
  def __init__(self, key: str, display_name: str, default_value: float, min_value: float, max_value: float, resolution: int = 0.1, places: int = 2):
    super().__init__()
    self._min_value = min_value
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

class ConfigInt(QtWidgets.QVBoxLayout):
  def __init__(self, key: str, display_name: str, default_value: int, min_value: int, max_value: int):
    super().__init__()
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

class ConfigEnum(QtWidgets.QHBoxLayout):
  def __init__(self, key: str, display_name: str, default_index: int, options: list[tuple[str, str]]):
    super().__init__()
    assert default_index >= 0 and default_index < len(options)
    self.addWidget(QtWidgets.QLabel(display_name))
    combobox = QtWidgets.QComboBox()
    for enum_option_display, enum_option_val in options:
      combobox.addItem(enum_option_display, enum_option_val)
    combobox.setCurrentIndex(default_index)
    self.addWidget(combobox)

@dataclass
class ConfigBlockDescription(ConfigBlockElement):
  text: str

@dataclass
class ConfigBlockSpec:
  block_name: str
  checkable: bool
  elements: Sequence[ConfigBlockElement]

class ConfigBlock(QtWidgets.QGroupBox):
  def __init__(self, config_block_spec: ConfigBlockSpec, parent: QtWidgets.QWidget = None):
    super().__init__(config_block_spec.block_name, parent)
    self.setCheckable(config_block_spec.checkable)

    v_layout = QtWidgets.QVBoxLayout(self)

    # Slots used in value updates within configs
    self._slots = []

    for element in config_block_spec.elements:
      if isinstance(element, ConfigBool):
        v_layout.addWidget(element)
      elif isinstance(element, ConfigInt):
        v_layout.addLayout(element)
      elif isinstance(element, ConfigFloat):
        v_layout.addLayout(element)
      elif isinstance(element, ConfigEnum):
        v_layout.addLayout(element)
      elif isinstance(element, ConfigBlockDescription):
        label = QtWidgets.QLabel(element.text)
        label.setWordWrap(True)
        v_layout.addWidget(label)
      else:
        raise ValueError(f'What do we do with a {option}?')
