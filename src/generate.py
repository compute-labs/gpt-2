import os
import json
import numpy as np
import tensorflow as tf
from src import model, sample, encoder

# Disable eager execution
tf.compat.v1.disable_eager_execution()

def generate_text(model_name='124M', models_dir='models', length=100, temperature=0.7, top_k=40, seed=None):
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    context = tf.compat.v1.placeholder(tf.int32, [1, None])
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    output = sample.sample_sequence(
        hparams=hparams, length=length,
        context=context,
        batch_size=1,
        temperature=temperature, top_k=top_k
    )

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session(config=config) as sess:
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        context_tokens = enc.encode("Once upon a time")
        out = sess.run(output, feed_dict={
            context: [context_tokens]
        })
        text = enc.decode(out[0])
        print(text)

if __name__ == '__main__':
    generate_text()

