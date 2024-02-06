from contextlib import redirect_stdout


def model_summary(model, folder_summary):
    """
    save the summary to txt
    """

    with open(folder_summary, 'w') as f:
        with redirect_stdout(f):
            model.summary()


def visualize_model(model, folder_plot, folder_summary):
    """
    from contextlib import redirect_stdout
    save the summary to txt,
    plot the model to png
    """
    from keras.utils import plot_model

    plot_model(model, to_file=folder_plot, show_shapes=True)
    model_summary(model, folder_summary)


def visualize_model_tf(model, folder_plot, folder_summary):
    """
    for models built with tf.keras use this function to visualize them
    from contextlib import redirect_stdout
    save the summary to txt,
    plot the model to png
    """
    import tensorflow as tf

    tf.keras.utils.plot_model(model, folder_plot, show_shapes=True)
    model_summary(model, folder_summary)
