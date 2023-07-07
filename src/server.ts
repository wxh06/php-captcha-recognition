import Fastify from "fastify";
import predict from "./predict.js";

const fastify = Fastify({
  logger: true,
});

// eslint-disable-next-line consistent-return
fastify.post("/predict/", (request, reply) => {
  if (
    typeof request.body !== "string" ||
    !request.body.startsWith("data:image/jpeg;base64,")
  )
    return reply.code(400).send("");
  predict(
    Buffer.from(request.body.slice("data:image/jpeg;base64,".length), "base64"),
  )
    .then((value) => reply.send(value))
    .catch((err: Error) => reply.code(500).send(err.message));
});

fastify
  .listen({ port: parseInt(process.env.PORT ?? "3000", 10) })
  .catch((err) => {
    fastify.log.error(err);
    process.exit(1);
  });
