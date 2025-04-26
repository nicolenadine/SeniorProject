import tensorflow as tf


class Trainer:
    def __init__(self, model, data_handler, output_dir, epochs):
        """
        Args:
            model (tf.keras.Model): The CNN model to train.
            data_handler (DataHandler): An instance of the DataHandler with train_files and train_labels.
            output_dir (str): Directory where outputs (e.g., logs, models) are saved.
            epochs (int): Number of epochs for training.
        """
        self.model = model
        self.data_handler = data_handler
        self.output_dir = output_dir
        self.epochs = epochs

        # Make sure datasets are prepared
        if not hasattr(self.data_handler, 'train_dataset') or self.data_handler.train_dataset is None:
            self.data_handler.setup_data_generators()

    def train(self, callbacks=None):
        """
        Trains the model using the training split provided by data_handler.

        Args:
            callbacks (list): List of TensorFlow callbacks.

        Returns:
            history: The training history returned by model.fit.
        """

        # Add a custom callback to monitor validation loss
        class LossMonitor(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                print(f"Epoch {epoch + 1} validation loss: {logs.get('val_loss'):.6f}")

        if callbacks is not None:
            callbacks.append(LossMonitor())

        # Fit the model using the data_handler's datasets
        history = self.model.fit(
            self.data_handler.train_dataset,
            epochs=self.epochs,
            validation_data=self.data_handler.val_dataset,
            callbacks=callbacks
        )

        return history