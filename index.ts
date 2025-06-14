import sharp from "sharp";
import * as tf from "@tensorflow/tfjs-node";

const CHARSET = "abcdefghijklmnpqrstuvwxyz123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

const vocabulary = Array.from(new Set(CHARSET.split(""))).sort();

class CaptchaSolver {
  private model: Promise<tf.GraphModel>;

  constructor(...args: Parameters<typeof tf.loadGraphModel>) {
    this.model = tf.loadGraphModel(...args);
  }

  async solve(images: Parameters<typeof sharp>[0][]): Promise<string[]> {
    const model = await this.model;

    const imageTensors = await Promise.all(
      images.map(async (image) => {
        const {
          data,
          info: { width, height, channels },
        } = await sharp(image).raw().toBuffer({ resolveWithObject: true });
        return tf.reshape(Array.from(data), [height, width, channels]);
      }),
    );

    const prediction = model.predict(
      tf.stack(imageTensors).cast("float32"),
    ) as tf.Tensor[];

    return Promise.all(
      prediction[0].unstack().map(async (t) =>
        Array.from(await t.data())
          .map((v) => vocabulary[v])
          .join(""),
      ),
    );
  }
}

export = CaptchaSolver;
