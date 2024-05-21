import os
import json
import numpy as np
import tensorflow as tf
from src import model, encoder

class HParams:
    n_vocab = 50257
    n_ctx = 1024
    n_embd = 768
    n_head = 12
    n_layer = 12

def load_dataset(enc, path):
    with open(path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    tokens = np.stack(enc.encode(raw_text))
    return tokens

def train(dataset, model_name='124M', models_dir='models', steps=1000, batch_size=1, learning_rate=0.0001):
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    context = tf.placeholder(tf.int32, [batch_size, None])
    output = model.model(hparams=hparams, X=context)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=context[:, 1:], logits=output['logits'][:, :-1]))

    train_vars = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss, var_list=train_vars)
    train_op = optimizer.apply_gradients(grads_and_vars)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(steps):
            np.random.shuffle(dataset)
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                feed_dict = {context: batch}
                _, loss_val = sess.run([train_op, loss], feed_dict=feed_dict)
                print(f'Step {step+1}/{steps}, Loss: {loss_val}')

if __name__ == '__main__':
    models_dir = 'models'  # Set the models directory
    dataset = load_dataset(encoder.get_encoder('124M', models_dir), 'dataset.txt')
    train(dataset, models_dir=models_dir)

