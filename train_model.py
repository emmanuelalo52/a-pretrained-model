from pretrained_data import *
#train the model
base_learning_rate=0.001#the rate at which you can modify the base model
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
              loss=tf.keras.losses.binary_crossentropy(from_logits=True),
              metric=['accuracy'])

#evaluate the model
inital_epochs = 3
validation_steps = 20

loss0,accuracy = model.evaluate(validation_batches,steps=validation_steps)

history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)
acc = history.history['accuracy']
print(acc)

#saving a model
model.save("dogs_vs_cats.h5")
new_model = tf.keras.models.load_model('dogs_vs_cats.h5')