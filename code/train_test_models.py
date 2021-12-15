import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import zipfile
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import pathlib
import pandas as pd
import pickle
import csv
import random
random.seed(10)
import time
from datetime import datetime
from datetime import timedelta
import shutil

import matplotlib.image as mpimg

print("\n" * 3)
delimeter = "*" * 100
print("*" * 50)
print("*" * 50)
print("%s Going to print if GPU is used" % delimeter)
print("%s Num GPUs Available: %d" % (delimeter, len(tf.config.list_physical_devices('GPU'))))
print(tf.config.list_physical_devices('GPU'))

# ============================================================== Utils 

def load_obj(base_dir, name):
    file_path = os.path.join(base_dir, "pickle", name + ".pkl")
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_obj(obj, base_dir, name):
    dir_path = os.path.join(base_dir, "pickle")
    create_directory(dir_path)
    file_path = os.path.join(dir_path, name + ".pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print("Saved object to a file: %s" % (str(file_path)))


def save_df(df, file_name, append=False):
    if append:
        df.to_csv(file_name, index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC, mode="a", header=False)
    else:
        df.to_csv(file_name, index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

def save_df(df, base_dir, file_name):
    file_name = os.path.join(base_dir, file_name)
    df.to_csv(file_name, index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

def save_df(df, file_name):
    df.to_csv(file_name, index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

def remove_directory(path):
    if os.path.exists(path):
        print("%s path exists and removing it." % path)
        shutil.rmtree(path)

def remove_file(file_name):
    if (os.path.isfile(file_name)):
        print("Output file %s exists and removing it." % file_name)
        os.remove(file_name)

def create_directory(dir):
    if(not os.path.exists(dir)):
        print("Creating directory %s." % dir)
        os.makedirs(dir)
    else:
        print("Directory %s already exists and so returning." % dir)

def remove_and_create_directory(dir):
    print("Going to REMOVE and CREATE directory: %s" % dir)
    remove_directory(dir)
    create_directory(dir)


def get_list_of_image_from_directory(dir):
    res = list(pathlib.Path(dir).glob("**/*.jpg"))
    res += list(pathlib.Path(dir).glob("**/*.png"))
    print("Total %d images found in directory %s" % (len(res), dir))
    return res




# ===================================================================================== Data Augmentation






def get_data_generator_from_directory(dir_path, is_test_data=False):
    # Data generator parameters
    gen_params = {"featurewise_center":False,\
                  "samplewise_center":False,\
                  "featurewise_std_normalization":False,\
                  "samplewise_std_normalization":False,\
                  "zca_whitening":False,\
                  "rotation_range":20,\
                  "width_shift_range":0.1,\
                  "height_shift_range":0.1, \
                  "shear_range":0.2, \
                  "zoom_range":0.1,\
                  "horizontal_flip":True,\
                  "vertical_flip":True}

   
    shuffle = True
    if(is_test_data):
      generator = ImageDataGenerator(preprocessing_function = tf.keras.applications.efficientnet.preprocess_input)
      shuffle = False
    else:
      # For training and validation dataset
      generator = ImageDataGenerator(**gen_params, preprocessing_function = tf.keras.applications.efficientnet.preprocess_input)

    data_generator = generator.flow_from_directory(
        directory = dir_path,
        target_size=(img_W, img_H),
        color_mode="rgb",
        classes= CLASS_NAMES,
        class_mode="categorical",
        batch_size=BS,
        shuffle=shuffle,
        seed=84,
        interpolation="nearest",
    )
    return data_generator




class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    # This will stop the training when validation accurary is equal or above the threashole, 100% by default
    def on_epoch_end(self, epoch, logs=None): 
        val_acc = logs["val_accuracy"]
        print("=========================> Epoch is finished and validation accuracy is: %.2f" % val_acc, flush=True)
        if val_acc >= self.threshold:
            print("Validation accurary is higher than the threadhold %.2f and so stopping training" % (self.threshold))
            self.model.stop_training = True

def get_callbacks(model_checkpoint_path, threadhold=1.0):
   

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 3)


    monitor_it = tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, monitor='val_loss',\
                                                verbose=0,save_best_only=True,\
                                                save_weights_only=False,\
                                                mode='min')

    def scheduler(epoch, lr):
        if epoch%10 == 0 and epoch!= 0:
            lr = lr/2
        return lr

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)

    threshold_callback = MyThresholdCallback(threshold=threadhold)

    return early_stop, monitor_it, lr_schedule, threshold_callback









# ============================================================== Model design  ======================================
def baseline_model_1():
    model = tf.keras.models.Sequential(name="Baseline_1VGG")
    model.add(tf.keras.layers.BatchNormalization(input_shape=INPUT_SHAPE))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    # model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax'))
    return model


def baseline_model_2():
    model = tf.keras.models.Sequential(name="Baseline_1VGG_dropout")
    model.add(tf.keras.layers.BatchNormalization(input_shape=INPUT_SHAPE))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(0.20))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.20))
    model.add(tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax'))
    return model



# def baseline_model():
#     model = tf.keras.models.Sequential(name="Baseline_CNN")
#     model.add(tf.keras.layers.BatchNormalization(input_shape=INPUT_SHAPE))
#     model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
#     model.add(tf.keras.layers.Dropout(0.25))
    
#     model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#     model.add(tf.keras.layers.Dropout(0.25))

#     model.add(tf.keras.layers.Flatten())
#     model.add(tf.keras.layers.Dense(128, activation='relu'))
#     model.add(tf.keras.layers.Dropout(0.5))
#     model.add(tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax'))
#     return model


#uses globabl variable num_classes and input_shape
def get_VGG19_model():
    model = tf.keras.applications.vgg19.VGG19(
      include_top=True,
      weights=None,
      input_tensor=None,
      input_shape=INPUT_SHAPE, 
      pooling=None, 
      classes=len(CLASS_NAMES),
      classifier_activation='softmax'
    )
    # print(model.summary())
    return model



def get_VGG19_transfer_learning_model():
    base_model = tf.keras.applications.vgg19.VGG19(
      include_top=False,
      weights='imagenet',
      input_shape=INPUT_SHAPE,
    )
    base_model.trainable = False

    # print(model.summary())
    x1 = base_model(base_model.input, training = False)
    x2 = tf.keras.layers.Flatten()(x1)
    out = tf.keras.layers.Dense(len(CLASS_NAMES),activation = 'softmax')(x2)
    model = tf.keras.Model(inputs = base_model.input, outputs=out)
    print(model.summary())
    return model


#uses globabl variable num_classes and input_shape
def get_Resnet50_model():
    model = tf.keras.applications.resnet50.ResNet50(
      include_top=True,
      weights=None,
      input_tensor=None,
      input_shape=INPUT_SHAPE, 
      pooling=None, 
      classes=len(CLASS_NAMES),
      classifier_activation='softmax'
    )
    # print(model.summary())
    return model

def get_Resnet50_transfer_learning_model():
    base_model = tf.keras.applications.resnet50.ResNet50(
      include_top=False,
      weights='imagenet',
      input_shape=INPUT_SHAPE,
    )
    base_model.trainable = False

    # print(model.summary())
    x1 = base_model(base_model.input, training = False)
    x2 = tf.keras.layers.Flatten()(x1)
    out = tf.keras.layers.Dense(len(CLASS_NAMES),activation = 'softmax')(x2)
    model = tf.keras.Model(inputs = base_model.input, outputs=out)
    print(model.summary())
    return model


def get_EfficientNet_model():
    model = tf.keras.applications.efficientnet.EfficientNetB7(
      include_top=True,
      weights=None,
      input_tensor=None,
      input_shape=INPUT_SHAPE, 
      pooling=None, 
      classes=len(CLASS_NAMES),
      classifier_activation='softmax'
    )
    # print(model.summary())
    return model

def get_EfficientNet_transfer_learning_model():
    base_model = tf.keras.applications.efficientnet.EfficientNetB7(
      include_top=False,
      weights='imagenet',
      input_shape=INPUT_SHAPE,
    )
    base_model.trainable = False

    # print(model.summary())
    x1 = base_model(base_model.input, training = False)
    x2 = tf.keras.layers.Flatten()(x1)
    out = tf.keras.layers.Dense(len(CLASS_NAMES),activation = 'softmax')(x2)
    model = tf.keras.Model(inputs = base_model.input, outputs=out)
    print(model.summary())
    return model




# ========================================================= Model Initialization =================================

def save_model_summary(model, file_name):
    with open(file_name, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print("Model summary has been saved to file: %s" % file_name)

def get_model(model_name):
    if(model_name == "Baseline_model_1"):
        return baseline_model_1()
    if(model_name == "Baseline_model_2"):
        return baseline_model_2()
    if(model_name == "VGG19"):
        return get_VGG19_model()
    if(model_name == "VGG19_transfer_learning"):
        return get_VGG19_transfer_learning_model()
    if(model_name == "Resnet50"):
        return get_Resnet50_model()
    if(model_name == "Resnet50_transfer_learning"):
        return get_Resnet50_transfer_learning_model()
    if(model_name == "EfficientNet"):
        return get_EfficientNet_model()
    if(model_name == "EfficientNet_transfer_learning"):
        return get_EfficientNet_transfer_learning_model()

# summarize history for accuracy
def plot_training_accuracy(history, file_name, title):
    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    fig.savefig(file_name)
    plt.show()
    plt.close()
    plt.cla()
    plt.clf()
    

# summarize history for accuracy
def plot_training_loss(history, file_name, title):
    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    fig.savefig(file_name)
    plt.show()
    plt.close()
    plt.cla()
    plt.clf()
    



class Result:
    def __init__(self, y_true, y_pred, path):
        self.y_true = y_true
        self.y_pred = y_pred
        self.path = path        

    def __str__(self):
        return "True Label: %s\nPrediction: %s\nPath: %s\n" % (self.y_true, self.y_pred, self.path)


# test_image_files = test_generator.filenames
def get_wrongly_predicted_sample_indexes(y_true, y_pred, files):
    wrong_samples = []
    for i in range(len(y_true)):
      if y_true[i] != y_pred[i]:
        res = Result(y_true[i], y_pred[i], files[i])
        wrong_samples.append(res)
    print(len(wrong_samples))
    # print(wrong_samples)
    return wrong_samples

def visualize_incorrectly_predicted_samples(incorrectly_predicted_samples, base_dir):
    nrows = 8
    ncols = 4

    fig = plt.gcf()
    fig.set_size_inches(ncols*4, nrows*4)
    fig.tight_layout()
    # fig.subplots_adjust(top=0.2)
    fig.suptitle("Randomly sampled incorrectly predicted samples", fontsize="x-large", y=1.01)

    for i in range(min(len(incorrectly_predicted_samples), nrows*ncols)):
        # file_name = test_images[index]
        res = incorrectly_predicted_samples[i]
        file_path = os.path.join(base_dir, res.path)
        img = mpimg.imread(file_path)
        sp = plt.subplot(nrows, ncols, i + 1)
        title = "%s" % (res)
        sp.set_title(title)
        sp.axis('Off')
        plt.imshow(img)
    plt.tight_layout()
    fig.savefig(incorrectly_predicted_samples_file, dpi=200)


# useas global variable train_generator
def get_true_and_pred_labels(y_pred_model):
    y_pred_indices = np.argmax(y_pred_model, axis=1)
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())

    y_true_classes = test_generator.classes
    y_true_labels = [ labels[k] for k in y_true_classes]

    y_pred_labels = [labels[k] for k in y_pred_indices]
    # print(y_pred_labels[:10])

    test_image_files = test_generator.filenames

    return y_true_labels, y_pred_labels, test_image_files


def save_to_csv(y_true_labels, y_pred_labels, test_image_files, output_file):
    df = pd.DataFrame()
    # abs_path = [os.path.join(test_dir, path) for path in test_image_files]
    df['sample']  = test_image_files
    df['true_label'] = y_true_labels
    df['predicted_label'] = y_pred_labels
    df['verdict'] = df['true_label'] == df['predicted_label']
    save_df(df, output_file)

def summary_output(text, file_name):
    with open(file_name, 'a') as f:
        print(text, file=f)

        # =================================== ==========================================================================================


# ======================================================== Configuration and Glabal Variables




current_dataset = sys.argv[1]
model_name = sys.argv[2]

img_W = 128
img_H = 128
BS = 128 # batch size
INPUT_SHAPE = (img_W, img_H, 3)
CLASS_NAMES = ["mask", "without_mask"]




BASE_DIR = "/home/mdabdullahal.alamin/alamin/face_mask"
if current_dataset == "real":
    data_dir = "/home/mdabdullahal.alamin/alamin/face_mask/dataset/RMFD/organized/"
if current_dataset == "simulated":
    data_dir = "/home/mdabdullahal.alamin/alamin/face_mask/dataset/masknet/organized/"
model_output_dir = os.path.join(BASE_DIR, "model_output", current_dataset)


train_dir = os.path.join(data_dir, "train/")
validation_dir = os.path.join(data_dir, "validation/")
test_dir = os.path.join(data_dir, "test/")


get_list_of_image_from_directory(train_dir)
get_list_of_image_from_directory(validation_dir)
get_list_of_image_from_directory(test_dir)
print("Over")




print("Base %s\n Data dir: %s\nTrain dir: %s\n Validation dir: %s\n Test dir: %s" % (BASE_DIR, data_dir, train_dir, validation_dir, test_dir))


train_generator = get_data_generator_from_directory(train_dir)
validation_generator = get_data_generator_from_directory(validation_dir)
test_generator = get_data_generator_from_directory(test_dir, is_test_data=True)





print("%s ===> Model name: %s" % (delimeter, model_name))
current_model_output_dir = os.path.join(model_output_dir, "batch_jobs", model_name)
print("\n\n\n =====================================> Current Model output dir: %s   ===============================" % (current_model_output_dir))

model_checkpoint_path = os.path.join(current_model_output_dir, model_name + ".h5")
model_summary_file = os.path.join(current_model_output_dir, "model_summary.txt")
training_accuracy_title = "%s Model Training Accuracy over time" % model_name
training_loss_title = "%s Model Training Loss over time" % model_name
training_accuracy_file = os.path.join(current_model_output_dir, "training_accurary.png")
training_loss_file = os.path.join(current_model_output_dir, "training_loss.png")

training_history_file = os.path.join(current_model_output_dir, "training_history.csv")
training_history_text = ""

test_summary_file = os.path.join(current_model_output_dir, "test_summary.txt")
all_test_prediction_file = os.path.join(current_model_output_dir, "all_test_prediction.csv")
incorrectly_predicted_samples_file = os.path.join(current_model_output_dir, "incorrectly_predicted_samples.png")

total_training_time = 0

# ================================================ Only during training =================

remove_and_create_directory(current_model_output_dir)
early_stop, monitor_it, lr_schedule, threshold_callback = get_callbacks(model_checkpoint_path)

model = get_model(model_name)
save_model_summary(model, model_summary_file)
model.summary()

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()
training_history = model.fit(train_generator, epochs=500, verbose = 0, \
                       callbacks = [early_stop, monitor_it, lr_schedule, threshold_callback], \
                       validation_data = (validation_generator))

end_time = time.time()
total_training_time = int(end_time - start_time)
print("Trining over", flush=True)



hist_df = pd.DataFrame(training_history.history)
training_history_text = str(hist_df)
print(hist_df)
save_df(hist_df, training_history_file)


plot_training_loss(training_history, training_loss_file, training_loss_title)
plot_training_accuracy(training_history, training_accuracy_file, training_accuracy_title)

# ========================================================= 



best_model = tf.keras.models.load_model(model_checkpoint_path)
loss, accuracy = best_model.evaluate(test_generator)
print("Loss: %.3f Accuracy: %.3f%%" % (loss, accuracy * 100))

y_pred_org = best_model.predict(test_generator)
y_true_labels, y_pred_labels, test_image_files = get_true_and_pred_labels(y_pred_org)
incorrectly_predicted_samples = get_wrongly_predicted_sample_indexes(y_true_labels, y_pred_labels, test_image_files)
save_to_csv(y_true_labels, y_pred_labels, test_image_files, output_file=all_test_prediction_file)
visualize_incorrectly_predicted_samples(incorrectly_predicted_samples, base_dir=test_dir)


final_summary_text = "Model Test dataset:\nAccuracy: %.3f%%\nLoss: %.3f\n" % ( accuracy * 100, loss)
final_summary_text += "Total training time %d seconds %.1f Hours\n\n" % (total_training_time, total_training_time/3600)
final_summary_text += "Training epoch stats: %s\n\n" % training_history_text
final_summary_text += "Incorrectly Predicted Samples:\n%s\n" % '\n'.join([sam.path for sam in incorrectly_predicted_samples])
print(final_summary_text)
summary_output(final_summary_text, test_summary_file)

