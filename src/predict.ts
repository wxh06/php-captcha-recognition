import tf from "@tensorflow/tfjs-node";
import sharp from "sharp";

const MODEL_PATH = "saved_model";

const model = await tf.node.loadSavedModel(MODEL_PATH);

export default async (image: Parameters<typeof sharp>[0]) =>
  String.fromCharCode(
    ...(await tf
      .argMax(
        model.predict(
          tf.expandDims(
            tf.div(
              tf.cast(
                tf.reshape(
                  Array.from(await sharp(image).raw().toBuffer()),
                  [35, 90, 3],
                ),
                "float32",
              ),
              255,
            ),
            0,
          ),
        ) as tf.Tensor<tf.Rank>,
        -1,
      )
      .data()),
  );
