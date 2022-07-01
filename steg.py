#!/usr/bin/env python3

import os
import argparse
import re
import numpy as np
import numpy.typing as npt
from itertools import zip_longest
from typing import List
from PIL import Image
from pathlib import Path


class ValidationError(ValueError):
  def __init__(self, message: str, error_attribute: str, *args):
    self.message = message
    self.error_attribute = error_attribute
    super().__init__(message, error_attribute, *args)


class ErrorCodes:

  PATH_DOES_NOT_EXIST = "PathDoesNotExist"
  NOT_A_FILE = "NotAFile"
  PATH_NOT_SPECIFIED = "PathIsNULL"
  WRONG_TEXT = "InaproperiateTextContent"
  WRONG_FILE_FORMAT = "WrongFileFormat"
  BYTES_ERROR = "BytesOverflow"
  WRONG_CHAR = "CharOutsideASCIIRange"


class Validator:
  @staticmethod
  def validate_image_path(cls, path: str, mode: str = "read"):
    if mode.lower() == "read":
      if not path:
        raise ValidationError(
          message=f"Path to input image not specified!",
          error_attribute=cls.PATH_NOT_SPECIFIED,
        )
      if not os.path.exists(path):
        raise ValidationError(
          f"Given path does not exist!", cls.PATH_DOES_NOT_EXIST
        )
      if not Path(path).is_file():
        raise ValidationError(
          f"Input path does not point to a file image!", cls.NOT_A_FILE
        )
      if not path.endswith(((".png", ".jpg", ".gif", ".jpeg", ".tiff", ".bmp"))):
        raise ValidationError(
          f"Image has wrong format. (`.png`, `.jpg`, `.gif`, `.jpeg`, `.tiff`, `.bmp` are correct ones!)",
          cls.WRONG_FILE_FORMAT,
        )
    else:
      ...

  @staticmethod
  def validate_bytes(cls, image: List[Image.Image], text: str):
    w, h = image.size
    if w * h < len(text):
      raise ValidationError(
          f"Text is too long: Available bytes: {w * h}, Image bytes: {len(text)}",
          cls.BYTES_ERROR,
      )

  # TODO: do validaiton on text image.
  @staticmethod
  def validate_text(cls, text: str):

    if any((not 0 <= ord(char) <= 255 for char in text)):
      raise ValueError(
        f"Text context is outside possible range from ascii table [0-255]!",
        cls.WRONG_CHAR,
      )


class ImageManager:
  __validator = Validator()

  def __init__(self, path: str, path_: str):
    # Validate the image path
    self.__validator.validate_image_path(ErrorCodes, path, mode="read")
    self.image_path = path
    self.__validator.validate_image_path(ErrorCodes, path_, mode="save")
    self.output_path = path_

  def __load_image(self) -> List[Image.Image]:
    return Image.open(self.image_path)

  def load_image(self) -> List[Image.Image]:
    return self.convert_image(self.__load_image(), mode="rgb")

  def save_image(self, input_image: List[Image.Image], rgb: list) -> None:
    output_image = Image.new(input_image.mode, input_image.size)
    output_image.putdata(rgb)
    output_image.save(self.output_path)

  def __convert(
      self, image: List[Image.Image], mode: str = "rgb"
  ) -> List[Image.Image]:
    output_mode = mode.upper()
    if image.mode == "P":
      converted_image = image.convert(mode=output_mode)
    if image.mode == "RGB" and image.has_transparency():
      converted_image = image.convert(mode=output_mode)

    return converted_image

  def convert_image(self, image: List[Image.Image], mode: str = "rgb"):
    return self.__convert(image, mode)


class TextLSB:
  """
  Steganography class for merge and unmerge images or hiding text within an image using LSB (Least significant bit).
  https://en.wikipedia.org/wiki/Steganography
"""

  validator = Validator()

  def encode(self, image: List[Image.Image], text: str):
    self.validator.validate_bytes(ErrorCodes, image, text)
    self.validator.validate_text(ErrorCodes, text)
    return self.__encode(image, text=text)

  def decode(image: List[Image.Image]):
    # TODO: Create decoding image
    ...

  def __encode(self, image: List[Image.Image], text: str):
    rgb_channels = self.__rgb_to_binary(image)
    ascii_codes = "".join(self.__ascii_to_binary(text=text))

    index = 0
    for count, (rgb_bin, ascii_bit) in enumerate(
      zip_longest(
          (bin_ for channel in rgb_channels for bin_ in channel), ascii_codes
      ),
      start=0,
    ):

      if count % 3 == 0 and count != 0:
        index += 1

      if ascii_bit != None:
        if ascii_bit == rgb_bin[-1]:
            continue

        lsb_rgb = re.sub(r".$", ascii_bit, rgb_bin)

        try:
          rgb_channel = list(rgb_channels[index])
          rgb_channel[rgb_channel.index(rgb_bin)] = lsb_rgb
          rgb_channels[index] = tuple(rgb_channel)
        except IndexError as e:
          print(e)
      else:
          break

    rgb_channels = list(map(self.__binary_to_int, rgb_channels))
    return rgb_channels

  def __binary_to_int(self, rgb: tuple) -> tuple:
    r, g, b = rgb
    return int(r, 2), int(g, 2), int(b, 2)

  def __int_to_binary(self, pixel: int) -> str:
    return f"{pixel:08b}"

  def __text_to_ascii(self, text: str):
    return list(map(ord, text))

  def __rgb_to_binary(self, image: List[Image.Image]):
    """
      R, G, B - each channel is converted to binary [[R channel in bin],
                                                     [G channel in bin],
                                                     [B channel in bin]]
    """
    image = list(image.getdata())
    bin_image = [tuple(map(self.__int_to_binary, channel)) for channel in image]
    return bin_image

  def __ascii_to_binary(self, text: str):
    bin_ascii = list(map(self.__int_to_binary, self.__text_to_ascii(text=text)))
    return bin_ascii


def main():
  """
  Parser is responsible for reading input image and saving the processed image in desirable destination path.
  Steganography tool uses two modes:
* --text which embeds text fragment into a picture using LSB algorithm (Least Significant Bit),
...
"""
  parser = argparse.ArgumentParser(description="Little tool for steganography.")
  parser.add_argument("--input", type=str, help="Input image destination")
  parser.add_argument(
      "--output", type=str, help="Destination folder to save processed image."
  )
  parser.add_argument(
      "--text", type=str, help="Text content to encode within the image."
  )
  args = parser.parse_args()

  if hasattr(args, "text") and args.text:
    manager = ImageManager(args.input, args.output)
    image = manager.load_image()
    rgb_list = TextLSB().encode(image, text=args.text)
    manager.save_image(image, rgb_list)
  else:
    print("...")


if __name__ == "__main__":
  main()
