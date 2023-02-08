from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from os import path
from subprocess import PIPE, Popen
from sys import argv
from uuid import uuid4

from numpy import array
from pandas import DataFrame
from PIL import Image

PHP_BIN = "php"
DATA_DIR = "data"
BUFFER_SIZE = 65536
IMAGES_PER_FILE = 4096


def generate():
    process = Popen([PHP_BIN, "generate.php"], stdout=PIPE)
    output = b""
    i = 0
    while True:
        output += process.stdout.read(BUFFER_SIZE)
        while True:
            eoi = output.find(b"\xff\xd9")
            if eoi < 0:
                break
            soi = output.find(b"\xff\xd8")
            eoi += 2
            i += 1
            if not i % 1024:
                print(i)

            phrase = output[:soi].decode("ascii")
            image = array(Image.open(BytesIO(output[soi:eoi])))
            output = output[eoi:]
            yield phrase, image

            if i >= IMAGES_PER_FILE:
                process.terminate()
                return


def save(df: DataFrame):
    uuid: str = uuid4().hex
    print(uuid)
    df.to_pickle(path.join(DATA_DIR, f"{uuid}.pkl"))


def main():
    save(DataFrame(generate(), columns=["phrase", "image"]))


if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(main) for i in range(int(argv[1]))]
