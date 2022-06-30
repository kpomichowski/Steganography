#!/usr/bin/env python3

import os
import argparse
import numpy as np
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


class Validator:
  @staticmethod
  def validate_image_path(cls, path: str):
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
              f"Image has wrong format. (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp` are correct ones!)",
              cls.WRONG_FILE_FORMAT,
          )

  # TODO: do validaiton on text image.
  @staticmethod
  def validate_text(self, text: str):
      ...


class ImageManager:
  __validator = Validator()

  def __init__(self, path: str):
      # Validate the image path
      self.__validator.validate_image_path(ErrorCodes, path)
      self.image_path = path

  def __load_image(self) -> List[Image.Image]:
      return Image.open(self.image_path)

  def load_image(self) -> List[Image.Image]:
      return self.__load_image()

  def __convert(
      self, image: List[Image.Image], mode: str = "rgb"
  ) -> List[Image.Image]:
      output_mode = mode.upper()
      if image.mode == "P":
          converted_image = image.convert(mode=output_mode)
      if image.mode == "RGB" and image.has_transparency():
          converted_image = image.convert(mode=output_mode)

      return converted_image

  def convert_image(self, image: List[Image.Image], mode: str = 'rgb'):
    return self.__convert(image, mode)


class Stega:
  """
    Steganography class for merge and unmerge images or hiding text within an image using LSB (Least significant bit).
    https://en.wikipedia.org/wiki/Steganography
  """

  def __init__(self, input_image_path: str, text: str = ""):
      self.input_manager = ImageManager(input_image_path)


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

  stega = Stega(input_image_path=args.input)
  image = stega.input_manager.load_image()
  print(stega.input_manager.convert_image(image, 'rgb').mode) 

if __name__ == "__main__":
  main()
