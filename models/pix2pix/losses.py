import tensorflow as tf
from tensorflow import losses

from models.pix2pix.hyperparameters import LAMBDA

loss_object = losses.BinaryCrossentropy(from_logits=True)
def GeneratorLoss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


class DiscriminatorLoss(losses.Loss):
    def __init__(self, name='disc_loss'):
        super(DiscriminatorLoss, self).__init__(name=name)
        self.loss_obj = losses.BinaryCrossentropy(from_logits=True)

    def call(self, y_true, y_pred, **kwargs):
        real_loss = self.loss_obj(tf.ones_like(y_true), y_true)
        fake_loss = self.loss_obj(tf.zeros_like(y_pred), y_pred)
        return real_loss + fake_loss

    def get_config(self):
        return super(DiscriminatorLoss, self).get_config()