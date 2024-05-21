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

def load_dataset(enc, path, seq_length):
    with open(path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    tokens = enc.encode(raw_text)
    # Split tokens into sequences of length `seq_length`
    dataset = [tokens[i:i + seq_length] for i in range(0, len(tokens) - seq_length + 1, seq_length)]
    return np.array(dataset)

def train(dataset, model_name='124M', models_dir='models', steps=1000, batch_size=1, seq_length=1024, learning_rate=0.0001):
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    context = tf.compat.v1.placeholder(tf.int32, [batch_size, seq_length])
    output = model.model(hparams=hparams, X=context)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=context[:, 1:], logits=output['logits'][:, :-1]))

    train_vars = tf.compat.v1.trainable_variables()
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss, var_list=train_vars)
    train_op = optimizer.apply_gradients(grads_and_vars)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for step in range(steps):
            np.random.shuffle(dataset)
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                if len(batch) < batch_size:
                    break  # Skip the last batch if it's smaller than `batch_size`
                feed_dict = {context: batch}
                _, loss_val = sess.run([train_op, loss], feed_dict=feed_dict)
                print(f'Step {step+1}/{steps}, Loss: {loss_val}')

            # Save the model every 100 steps
            if (step + 1) % 100 == 0:
                saver.save(sess, os.path.join(models_dir, model_name, 'model.ckpt'), global_step=step+1)
        
        # Save the final model
        saver.save(sess, os.path.join(models_dir, model_name, 'model.ckpt'), global_step=steps)

if __name__ == '__main__':
    models_dir = 'models'  # Set the models directory
    seq_length = 1024  # Define the sequence length
    dataset = load_dataset(encoder.get_encoder('124M', models_dir), 'dataset.txt', seq_length)
    train(dataset, models_dir=models_dir, seq_length=seq_length)

