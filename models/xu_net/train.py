from tensorflow import losses, optimizers, metrics, keras

from models.xu_net.LearningRateDecay import scheduler
from models.xu_net.XuNet import XuNet
from models.xu_net.data_process import X_train, y_train, X_test, y_test
from models.xu_net.hyperparameters import EPOCHS, BATCH_SIZE

if __name__ == '__main__':

    optimizer = optimizers.Adam(learning_rate=1e-4)
    xu_net = XuNet()
    callbacks = [keras.callbacks.LearningRateScheduler(scheduler)]
    xu_net.compile(loss=losses.categorical_crossentropy, optimizer=optimizer, metrics=[metrics.AUC()])
    xu_net.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)
    xu_net.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)