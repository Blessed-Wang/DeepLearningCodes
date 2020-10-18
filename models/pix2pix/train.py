import pprint
import time

import tensorflow as tf
import datetime
import os
from IPython import display
from tensorflow import optimizers
from data.load_facades import PATH
from models.pix2pix.Discriminator import Discriminator
from models.pix2pix.Generator import Generator
from models.pix2pix.data_process import load_image_train, load_image_test
from models.pix2pix.hyperparameters import BUFFER_SIZE, BATCH_SIZE, OUTPUT_CHANNELS, LAMBDA, EPOCHS
from models.pix2pix.losses import GeneratorLoss, DiscriminatorLoss
from models.pix2pix.utils import generate_images

if __name__ == '__main__':
    train_dataset = tf.data.Dataset.list_files(PATH + 'train/*.jpg')
    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.list_files(PATH + 'test/*.jpg')
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    generator = Generator(output_channels=OUTPUT_CHANNELS, name='generator')
    discriminator = Discriminator()
    generator_loss = GeneratorLoss
    discriminator_loss = DiscriminatorLoss()
    generator_optimizer = optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
    discriminator_optimizer = optimizers.Adam(learning_rate=1e-4, beta_1=0.5)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer,
                                     generator=generator, discriminator=discriminator)

    log_dir = "logs/"
    summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    @tf.function
    def train_step(input_image, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)
            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)


    def fit(train_ds, epochs, test_ds):
        for epoch in range(epochs):
            start = time.time()
            display.clear_output(wait=True)
            for example_input, example_target in test_ds.take(1):
                generate_images(generator, example_input, example_target)
            print("Epoch: ", epoch)
            # Train
            for n, (input_image, target) in train_ds.enumerate():
                print('.', end='')
                if (n + 1) % 100 == 0:
                    print()
                train_step(input_image, target, epoch)
            print()
            # saving (checkpoint) the model every 20 epochs
            if (epoch + 1) % 20 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))
        checkpoint.save(file_prefix=checkpoint_prefix)


    fit(train_dataset, EPOCHS, test_dataset)