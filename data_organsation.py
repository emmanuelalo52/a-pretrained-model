from data_prep import *
get_label_name = metadata.features['label'].int2str#creates a funstion that can be used as labels
for image,label in raw_train.take(2):
  plt.figure()
  plt.imshow()
  plt.title(get_label_name)

IMG_SIZE=160
def format_example(image,label):
  """
  returns the image reshaped
  """
  image = tf.cast(image,tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image,(IMG_SIZE,IMG_SIZE))
  return image,label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

for image,label in train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name)

for img,label in raw_train.take(2):
  print("original shape:",img.shape)
  for img,label in train.take(2):
    print("new shape", img.shape)