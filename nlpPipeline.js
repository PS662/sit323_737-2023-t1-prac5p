const tf = require("@tensorflow/tfjs-node");

const use = require("@tensorflow-models/universal-sentence-encoder");

async function loadModel() {
  const model = await use.load();
  return model;
}

async function computeEmbeddings(model, texts) {
  const embeddings = await model.embed(texts);
  return embeddings.arraySync();
}

function cosineSimilarity(a, b) {
  a = a.cast("float32");
  b = b.cast("float32");
  const dotProduct = a.mul(b).sum(1);
  const normA = a.norm(2, 1);
  const normB = b.norm(2, 1);
  const similarity = dotProduct.div(normA.mul(normB));
  return similarity.arraySync();
}

async function processQuery(userQuery, hotelAnnotations) {
  const model = await loadModel();
  const allTexts = [userQuery, ...hotelAnnotations];
  const embeddings = await computeEmbeddings(model, allTexts);
  const userEmbeddings = embeddings[0];
  const hotelEmbeddings = embeddings.slice(1);
  const similarities = cosineSimilarity(tf.tensor2d([userEmbeddings]), tf.tensor2d(hotelEmbeddings));
  return similarities[0];
}

module.exports = { processQuery };