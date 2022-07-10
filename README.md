# [Steganography](https://en.wikipedia.org/wiki/Steganography) - little tool for hiding text message within an image!

### Guess! Which image has been encoded?

Image #1 | Image #2
--- | ---
![](/images/encoded_lena.png) | ![](/images/lena.png)

<small>The answer is... **#Image #1**!</small>

### Purpose 

Command line tool for encoding/decoding the message within the pixels of the Image with usage the [LSB algorithm](https://en.wikipedia.org/wiki/Bit_numbering#Bit_significance_and_indexing).

### Built with Python 3.8.13.

# Usage 

# Output of the helper panel `python steg.py -h`:

```

usage: steg.py [-h] [--input INPUT] [--output OUTPUT] [--text TEXT] [--extract]

Little tool for steganography.

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    Input image destination
  --output OUTPUT  Destination folder to save processed image.
  --text TEXT      Encode text content within an image.
  --extract        Extract embedded text within an image.
  
```

### Usecase #1 - encoding the text message within the image:

Command example:

```
./steg.py --input images/lena.png --output . --text "Hello! This is an encrypted message."
```

Input `--input` specifies the image, to which the text message will be encoded. 

Output `--output` image can be relative path to the existing folder such as current path `.`, then the output image with embedded text will be named randomly, using `secrets.token_hex(8)`.

If the above command has been executed without an error, the following command would have appear:

`Succesfully saved an image to /home/user/folder/Steganography/12e4b0b4.png`

Note that the image can be saved to an existing path with whatever filename you like:

```
./steg.py --input images/lena.png --output ./test.png --text "Another example with custom filename of the decoded image!"
```

There will be always an information of the path to the saved image:

`Succesfully saved an image to /home/kp/Projects/Steganography/test.png`

### Usecase #2 - decoding the text from an image:

Command example:

```
python steg.py --input 12e4b0b4.png --extract
```

If the above command has been executed successfully, the following command would have appear:

```

      [2022-07-11 00:27:41] Decoding image content.
      [INFO] Image size: (512, 512); possible encoded bytes: 98304.
      [INFO] Decoded text:
      
      ```
        Hello! This is an encrypted message.

      ```
```


