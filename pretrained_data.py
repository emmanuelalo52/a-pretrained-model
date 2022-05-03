

#picking a pretained model
IMG_SHAPE=(IMG_SIZE,IMG_SIZE,3)
#the base model from the pretrained model
base_model=tf.keras.application.MobielNetV2(input_shape=IMG_SHAPE,
                                            include_top=False,
                                            weights='imagnet')#do we need the classification on the pretrained moddel?

for image,_ in train_batches.take(1):
  pass
feature_batch = base_model(image)
print(feature_batch.shape)

#freeze the training the base model
base_model.trainable=False

#developing our classifier
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer=keras.layers.Dense()

model = tf.keras.Sequential(
    base_model,
    global_average_layer,
    prediction_layer
)

model.summary()