import os
import time
import tensorflow as tf
from IPython import display

from models.dcgan import dcgan_model
from models.dcgan.hyperparameters import BUFFER_SIZE, BATCH_SIZE, EPOCHS, SEED, NOISE_DIM
from models.dcgan.losses import generator_loss, discriminator_loss
from models.dcgan.utils import generate_and_save_images

if __name__ == '__main__':
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # 将图片标准化到 [-1, 1] 区间内
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    generator = dcgan_model.Generator()
    discriminator = dcgan_model.Discriminator()

    generator_optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    discriminator_optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)


    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)
            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


    def train(dataset, epochs):
        for epoch in range(epochs):
            start = time.time()
            for image_batch in dataset:
                train_step(image_batch)
            # 继续进行时为 GIF 生成图像
            display.clear_output(wait=True)
            generate_and_save_images(generator, epoch + 1, SEED)
            # 每 15 个 epoch 保存一次模型
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            # 最后一个 epoch 结束后生成图片
            display.clear_output(wait=True)
            generate_and_save_images(generator, epoch, SEED)
    train(train_dataset, EPOCHS)