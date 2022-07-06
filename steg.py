#!/usr/bin/env python3

import os
import argparse
import datetime
import itertools
import time
import numpy as np
import numpy.typing as npt
import secrets
from typing import List, Tuple
from PIL import Image
from pathlib import Path


def print_decoded_message(image: List[Image.Image], decoded_string: str) -> None:
  print(
    f"""
      [{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Decoding image content.
      [INFO] Image size: {image.size}; possible encoded bytes: {(image.size[0] * image.size[1]) * 3 // 8}.
      [INFO] Decoded text:
      \t
      ```
        {decoded_string}

      ```
    """
  )


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
  INAPROPERIATE_SIZE = "WrongImageSize"


class Validator:
  @staticmethod
  def validate_image_path(cls, path: str, mode: str = "read"):

    if not path:
      raise ValidationError(
        message=f"Path to input image not specified!",
        error_attribute=cls.PATH_NOT_SPECIFIED,
      )

    if not os.path.exists(path):
      ext = os.path.splitext(path)[-1].lower()
      if ext not in [".tiff", ".png", ".jpg", ".jpeg"]:
        raise ValidationError(f"Given path does not exist!", cls.PATH_DOES_NOT_EXIST)

    if mode.lower() == "read":
      if not Path(path).is_file():
        raise ValidationError(
          f"Input path does not point to a file image!", cls.NOT_A_FILE
        )
      if not path.endswith(((".png", ".jpg", ".gif", ".jpeg", ".tiff", ".bmp"))):
        raise ValidationError(
          f"Wrong format of an input image. Available extensions: (`.png`, `.jpg`, `.gif`, `.jpeg`, `.tiff`, `.bmp` are correct ones!)",
          cls.WRONG_FILE_FORMAT,
        )
    else:
      if not Path(path).is_file() and Path(path).is_dir():
        filename = secrets.token_hex(nbytes=4)
        return filename
      elif not path.endswith((".png", ".jpg", ".jpeg", ".tiff")):
        raise ValidationError(
          f"Wrong format of an output image. Available extensions: (`.png`, `.jpg`, `.jpeg`, `.tiff`).",
          cls.WRONG_FILE_FORMAT,
        )

  @staticmethod
  def validate_bytes(cls, image: List[Image.Image], text: str):
    w, h = image.size
    if (w * h) * 3 // 8 < len(text):
      raise ValidationError(
        f"Text is too long: Available bytes: {(w * h) * 3 // 8}, Image bytes: {len(text)}",
        cls.BYTES_ERROR,
      )

  @staticmethod
  def validate_text(cls, text: str):

    if any((not 0 <= ord(char) <= 255 for char in text)):
      raise ValidationError(
        f"Text context is outside possible range from ascii table [0-255]!",
        cls.WRONG_CHAR,
      )

  @staticmethod
  def validate_image_size(
    cls, input_image: List[Image.Image], _input_image: List[Image.Image]
  ):
    if input_image.size < _input_image.size:
      raise ValidationError(
        f"Image to be merged must be smaller than the input image. Input size: {input_image.size}, Merge image: {_input_image.size}.",
        cls.INAPROPERIATE_SIZE,
      )


class ImageManager:
  __validator = Validator()

  def __init__(self, path: str):
    # Validate the image path
    self.__validator.validate_image_path(ErrorCodes, path, mode="read")
    self.image_path = path

  def __load_image(self) -> List[Image.Image]:
    return Image.open(self.image_path)

  def load_image(self) -> List[Image.Image]:
    image = self.__load_image()
    if image.mode != "RGB":
      image = self.convert_image(image, mode="rgb")
    return image

  def save_image(self, path: str, input_image: List[Image.Image], rgb: list) -> None:
    filename = self.__validator.validate_image_path(ErrorCodes, path, mode="save")
    output_image = Image.new(input_image.mode, input_image.size)
    output_image.putdata(rgb)
    DST_PATH = (
      os.path.join(path, filename + os.path.splitext(self.image_path)[-1])
      if filename
      else ""
    )
    output_image.save(path) if not filename else output_image.save(DST_PATH)
    self.output_image = path if not DST_PATH else DST_PATH

  def check_save_file(self) -> bool:
    return Path(self.output_image).is_file()

  def __convert(self, image: List[Image.Image], mode: str = "rgb") -> List[Image.Image]:
    output_mode = mode.upper()

    converted_image = None
    if image.mode == "P":
      converted_image = image.convert(mode=output_mode)
    if image.mode == "RGB" and not image.info.get("transparency"):
      converted_image = image.convert(mode=output_mode)

    return image if not converted_image else converted_image

  def convert_image(self, image: List[Image.Image], mode: str = "rgb"):
    return self.__convert(image, mode)


def print_saved_image_path(manager: ImageManager, path: str) -> None:
  if manager.check_save_file():
    print(f"Succesfully saved an image to {os.path.abspath(manager.output_image)}")
  else:
    print(f"Error saving an image to {os.path.abspath(os.path.abspath(path))}")


class TextLSB:
  """
    Steganography class for merge and unmerge images or hiding text within an image using LSB (Least significant bit).
    https://en.wikipedia.org/wiki/Steganography
    """

  __validator = Validator()

  def encode(self, image: List[Image.Image], text: str):
    self.__validator.validate_bytes(ErrorCodes, image, text)
    self.__validator.validate_text(ErrorCodes, text)
    return self.__encode(image, text=text)

  def __encode(self, image: List[Image.Image], text: str):
    rgb_channels = self.__rgb_to_binary(image)
    ascii_codes = "".join(self.__ascii_to_binary(text=text))

    lsb_bits = []
    chunk_size = 3
    bin_ascii_substrings = [
      ascii_codes[y - chunk_size : y]
      for y in range(chunk_size, len(ascii_codes) + chunk_size, chunk_size)
      if ascii_codes[y - chunk_size : y] != ""
    ]

    for bin_ascii, channel in itertools.zip_longest(bin_ascii_substrings, rgb_channels):
      if bin_ascii and channel:
        channel_copy = list(channel)
        for i in range(len(bin_ascii)):
          if bin_ascii[i]:
            channel_copy[i] = channel[i][:-1] + bin_ascii[i]
        lsb_bits.append(tuple(channel_copy))
      else:
        lsb_bits.append(channel)

    rgb_channels = list(map(self.__binary_to_int, lsb_bits))
    return rgb_channels

  def decode(self, image: List[Image.Image]):
    decoded_message = self.__decode(image)
    return decoded_message

  def __decode(self, image: List[Image.Image]):
    rgb_binary = self.__rgb_to_binary(image)
    binary = "".join((binary[-1] for channel in rgb_binary for binary in channel))
    chunks, chunk_size = len(binary), 8
    bytes_ = [binary[i : i + chunk_size] for i in range(0, chunks, chunk_size)]
    decoded_string = "".join([chr(int(byte, 2)) for byte in bytes_]).partition("-----")[
      0
    ]
    return decoded_string

  def __binary_to_int(self, rgb: tuple) -> tuple:
    r, g, b = rgb
    return int(r, 2), int(g, 2), int(b, 2)

  def __int_to_binary(self, pixel: int) -> str:
    return f"{pixel:08b}"

  def __text_to_ascii(self, text: str):
    # specified deilimeter
    text += "-----"
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


class ImageMergeSteg:

  __validator = Validator()

  def __int_to_binary(self, rgb_pixels):
    r, g, b = rgb_pixels
    return int(r, 2), int(g, 2), int(b, 2)

  def __binary_to_int(self, rgb_pixels):
    r, g, b = rgb_pixels
    return f"{r:08b}, {g:08b}, {b:08b}"

  def merge(
    self, input_image: List[Image.Image], merge_image: List[Image.Image]
  ) -> List[Image.Image]:
    return self.__merge(input_image, merge_image)

  def __merge(
    self, input_image: List[Image.Image], merge_image: List[Image.Image]
  ) -> List[Image.Image]:
    return ...


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
  parser.add_argument("--text", type=str, help="Encode text content within an image.")
  # decoding text from an image
  parser.add_argument(
    "--extract", action="store_true", help="Extract embedded text within an image."
  )
  parser.add_argument(
    "--merge", type=str, help="A specific image to be encoded into the input image."
  )

  args = parser.parse_args()

  manager = ImageManager(args.input)
  image = manager.load_image()

  if hasattr(args, "text") and args.text:
    rgb_list = TextLSB().encode(image, text=args.text)
    manager.save_image(args.output, image, rgb_list)
    print_saved_image_path(manager, args.output)
  elif hasattr(args, "extract") and args.extract:
    decoded_text = TextLSB().decode(image)
    print_decoded_message(image, decoded_text)
  elif hasattr(args, "merge") and args.merge:
    image_to_be_merged = ImageManager(args.merge).load_image()
    rgb_list = ImageMergeSteg().merge(image, image_to_be_merged)
    manager.save_image(args.output, image, rgb_list)
    print_saved_image_path(manager, args.output)


if __name__ == "__main__":
  main()
