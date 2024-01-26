# php-captcha-recognition

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Train OCR model for reading [Gregwar/Captcha](https://github.com/Gregwar/Captcha)

## Python dependencies

```sh
pip install -r requirements.txt
```

## PHP dependencies

### Ubuntu

```sh
sudo apt-get install php php-curl php-gd php-mbstring

php -r "copy('https://getcomposer.org/installer', 'composer-setup.php');"
php composer-setup.php
php -r "unlink('composer-setup.php');"

php composer.phar update
```

### macOS

```sh
brew install php composer

composer update
```

## Generating data

```bash
for _ in {0..3}; do uuidgen; done | parallel --termseq INT python3 generate.py
```

## Training

```sh
python3 train.py
```
