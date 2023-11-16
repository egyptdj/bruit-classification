import os
import json
import glob
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

YAMNET_HANDLE = 'https://tfhub.dev/google/yamnet/1'
SAMPLE_RATE = 16000
INPUT_SECONDS = None
INPUT_LENGTH = 15600
BATCH_SIZE = 1
SEED = 0

if INPUT_SECONDS:
  INPUT_LENGTH = int(SAMPLE_RATE * INPUT_SECONDS)

BASEDIR = '/Users/egyptdj/github/bruit-classification'
DATADIR = os.path.join(BASEDIR, 'data')
MODELDIR = os.path.join(BASEDIR, 'models', f'twostep_model_length{INPUT_LENGTH}')
ONESTEP_MODELDIR = os.path.join(BASEDIR, 'models', f'onestep_model_length{INPUT_LENGTH}')
os.makedirs(MODELDIR, exist_ok=True)

if __name__ == '__main__':
  random.seed(SEED)
  np.random.seed(SEED)
  tf.random.set_seed(SEED)
  tf.keras.utils.set_random_seed(SEED)
  tf.config.experimental.enable_op_determinism()


  train_audio_file_list = pd.Series(glob.glob(os.path.join(DATADIR, 'train', '**/*.wav')), dtype='object')
  train_audio_label_list = [f.split('/')[-2] for f in train_audio_file_list]


  val_audio_file_list = pd.Series(glob.glob(os.path.join(DATADIR, 'val', '**/*.wav')), dtype='object')
  val_audio_label_list = [f.split('/')[-2] for f in val_audio_file_list]


  test_audio_file_list = pd.Series(glob.glob(os.path.join(DATADIR, 'test', '**/*.wav')), dtype='object')
  test_audio_label_list = [f.split('/')[-2] for f in test_audio_file_list]


  class_name_dict = {}
  i = 0
  for label in train_audio_label_list:
    if not label in list(class_name_dict.keys()):
      class_name_dict[label] = i
      i += 1


  with open(os.path.join(MODELDIR, 'class_name_dict.json'), 'w') as f:
    json.dump(class_name_dict, f)


  train_audio_label_list = pd.Series([class_name_dict[i] for i in train_audio_label_list], dtype='int64')
  val_audio_label_list = pd.Series([class_name_dict[i] for i in val_audio_label_list], dtype='int64')
  test_audio_label_list = pd.Series([class_name_dict[i] for i in test_audio_label_list], dtype='int64')


  train_audio_label_list_step1 = train_audio_label_list.map(lambda x: 0 if x==class_name_dict['background'] else 1)
  val_audio_label_list_step1 = val_audio_label_list.map(lambda x: 0 if x==class_name_dict['background'] else 1)
  test_audio_label_list_step1 = test_audio_label_list.map(lambda x: 0 if x==class_name_dict['background'] else 1)


  train_audio_label_list_step2 = train_audio_label_list.replace({class_name_dict['background']: -1, class_name_dict['normal']: 0, class_name_dict['bruit']: 1, class_name_dict['normal-noise']: 2, class_name_dict['bruit-noise']: 2})
  val_audio_label_list_step2 = val_audio_label_list.replace({class_name_dict['background']: -1, class_name_dict['normal']: 0, class_name_dict['bruit']: 1, class_name_dict['normal-noise']: 2, class_name_dict['bruit-noise']: 2})
  test_audio_label_list_step2 = test_audio_label_list.replace({class_name_dict['background']: -1, class_name_dict['normal']: 0, class_name_dict['bruit']: 1, class_name_dict['normal-noise']: 2, class_name_dict['bruit-noise']: 2})


  train_ds1 = tf.data.Dataset.from_tensor_slices((train_audio_file_list, train_audio_label_list_step1))
  val_ds1 = tf.data.Dataset.from_tensor_slices((val_audio_file_list, val_audio_label_list_step1))
  test_ds1 = tf.data.Dataset.from_tensor_slices((test_audio_file_list, test_audio_label_list_step1))


  train_ds2 = tf.data.Dataset.from_tensor_slices((train_audio_file_list, train_audio_label_list_step2)).filter(lambda x, y: tf.math.logical_not(tf.math.equal(y, -1)))
  val_ds2 = tf.data.Dataset.from_tensor_slices((val_audio_file_list, val_audio_label_list_step2)).filter(lambda x, y: tf.math.logical_not(tf.math.equal(y, -1)))
  test_ds2 = tf.data.Dataset.from_tensor_slices((test_audio_file_list, test_audio_label_list_step2)).filter(lambda x, y: tf.math.logical_not(tf.math.equal(y, -1)))


  # *.wav 파일을 경로에서부터 오디오파일로 읽어오기

  @tf.function
  def load_wav_and_label(audio_file, label):
    audio = tf.io.read_file(audio_file)
    audio, sample_rate = tf.audio.decode_wav(audio)
    audio = tf.squeeze(audio, axis=-1)
    return audio, label


  @tf.function
  def slide_window(audio, window_size=15600, stride=7800):
    return [audio[...,w:w+window_size] for w in range(0, audio.shape[-1]-window_size, stride)]


  @tf.function
  def convert_label_to_onehot(audio, label, num_classes=3):
    return audio, tf.one_hot(label, num_classes)


  yamnet_model = hub.load(YAMNET_HANDLE)


  @tf.function
  def extract_embedding(wav_data, label):
    scores, embeddings, spectrogram = yamnet_model(wav_data)
    num_embeddings = tf.shape(embeddings)[0]
    return embeddings, tf.repeat(label, num_embeddings)


  model_step1 = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(1024), dtype=tf.float32, name='yamnet_embedding_step1'),
      tf.keras.layers.Dense(1, activation='sigmoid'),
  ])

  model_step2 = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(1024), dtype=tf.float32, name='yamnet_embedding_step2'),
      tf.keras.layers.Dense(3),
      tf.keras.layers.Softmax(),
  ])

  model_step1.summary()
  model_step2.summary()


  metrics = ['accuracy', 
            tf.keras.metrics.AUC(), 
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
            ]

  model_step1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=1e-5,
                ),
                metrics=metrics,
  )


  model_step2.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=1e-5,
                ),
                metrics=metrics,
  )


  callbacks = [tf.keras.callbacks.EarlyStopping(
                  monitor='loss',
                  patience=3,
                  restore_best_weights=True
              ),
              tf.keras.callbacks.LearningRateScheduler(
                tf.keras.optimizers.schedules.CosineDecay(
                  initial_learning_rate=1e-5,
                  decay_steps=100,
                )
              )
  ]

  train_ds1 = train_ds1.map(load_wav_and_label).map(extract_embedding).cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
  val_ds1 = val_ds1.map(load_wav_and_label).map(extract_embedding).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
  test_ds1 = test_ds1.map(load_wav_and_label).map(extract_embedding).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

  train_ds2 = train_ds2.map(load_wav_and_label).map(extract_embedding).map(convert_label_to_onehot).cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
  val_ds2 = val_ds2.map(load_wav_and_label).map(extract_embedding).map(convert_label_to_onehot).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
  test_ds2 = test_ds2.map(load_wav_and_label).map(extract_embedding).map(convert_label_to_onehot).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


  history1 = model_step1.fit(train_ds1,
                      epochs=100,
                      validation_data=val_ds1,
                      callbacks=callbacks
                      )

  model_step1.evaluate(test_ds1)

  history2 = model_step2.fit(train_ds2,
                      epochs=100,
                      validation_data=val_ds2,
                      callbacks=callbacks
                      )

  model_step2.evaluate(test_ds2)

  embedding_extraction_layer = hub.KerasLayer(YAMNET_HANDLE,
                                              input_shape=(INPUT_LENGTH,),
                                              trainable=False, name='yamnet')
  
  train_history1 = pd.DataFrame.from_dict({
    'Loss': history1.history['loss'], 
    'Accuracy': history1.history['accuracy'], 
    'Precision': history1.history['precision'], 
    'Recall': history1.history['recall'],
    'Split': ['Train' for _ in history1.history['loss']],
    'Epoch': [i+1 for i, _ in enumerate(history1.history['loss'])],
  })
  val_history1 = pd.DataFrame.from_dict({
    'Loss': history1.history['val_loss'], 
    'Accuracy': history1.history['val_accuracy'], 
    'Precision': history1.history['val_precision'], 
    'Recall': history1.history['val_recall'], 
    'Split': ['Validation' for _ in history1.history['val_loss']],
    'Epoch': [i+1 for i, _ in enumerate(history1.history['val_loss'])],
  })
  history1_df = pd.concat([train_history1, val_history1])

  train_history2 = pd.DataFrame.from_dict({
    'Loss': history2.history['loss'], 
    'Accuracy': history2.history['accuracy'], 
    'Precision': history2.history['precision'], 
    'Recall': history2.history['recall'],
    'Split': ['Train' for _ in history2.history['loss']],
    'Epoch': [i+1 for i, _ in enumerate(history2.history['loss'])],
  })
  val_history2 = pd.DataFrame.from_dict({
    'Loss': history2.history['val_loss'], 
    'Accuracy': history2.history['val_accuracy'], 
    'Precision': history2.history['val_precision'], 
    'Recall': history2.history['val_recall'], 
    'Split': ['Validation' for _ in history2.history['val_loss']],
    'Epoch': [i+1 for i, _ in enumerate(history2.history['val_loss'])],
  })
  history2_df = pd.concat([train_history2, val_history2])
  
  sns.set_theme(context='paper', style='whitegrid', font='helvetica', font_scale=1.5, palette='muted')
  plt.rc('text', usetex=True)

  fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(12,7), sharex=True)
  for i, metric in enumerate(['Loss', 'Accuracy', 'Precision', 'Recall']):
    h2 = sns.lineplot(history2_df, x='Epoch', y=metric, hue='Split', linewidth=2, legend=False, ax=ax[0][i])
    h1 = sns.lineplot(history1_df, x='Epoch', y=metric, hue='Split', linewidth=2, legend=True if i==3 else False, ax=ax[1][i])
    h1.set(ylabel='Carotic sound recognizer $g$' if i==0 else None, xlabel=None, title=metric)
    h2.set(ylabel='Bruit classifier $f$' if i==0 else None)
  plt.suptitle('Traning curve')
  plt.tight_layout()
  plt.savefig(os.path.join(MODELDIR, 'traincurve.png'))
  plt.close
  
  
  input_segment = tf.keras.layers.Input(batch_size=1, shape=(INPUT_LENGTH, ), dtype=tf.float32, name='audio')
  input_segment_reshaped = tf.reshape(input_segment, (INPUT_LENGTH,), name='reshape')
  _, embeddings, _ = embedding_extraction_layer(input_segment_reshaped)
  # step1_output = tf.cast(tf.argmax(tf.math.reduce_mean(model_step1(embeddings), axis=0), axis=0), tf.float32)
  step1_output = tf.sigmoid(tf.math.reduce_mean(model_step1(embeddings), axis=0))
  step1_output = tf.cond(step1_output > 0.5, lambda: tf.constant(1.0), lambda: tf.constant(0.0))
  step2_output = tf.math.reduce_mean(model_step2(embeddings), axis=0)
  step2_output = tf.nn.softmax(step2_output)
  final_output = tf.multiply(step1_output, step2_output, name='final_output')

  serving_model = tf.keras.Model(input_segment, final_output, name='bruit_model_twostep')
  serving_model.save(MODELDIR, include_optimizer=False)

  converter = tf.lite.TFLiteConverter.from_saved_model(MODELDIR)

  tflite_model = converter.convert()

  with open(os.path.join(MODELDIR, 'bruit_yamnet_twostep.tflite'), 'wb') as f:
    f.write(tflite_model)
    

  ## Onestep model
  
  input_segment = tf.keras.layers.Input(batch_size=1, shape=(INPUT_LENGTH, ), dtype=tf.float32, name='audio')
  input_segment_reshaped = tf.reshape(input_segment, (INPUT_LENGTH,), name='reshape')
  _, embeddings, _ = embedding_extraction_layer(input_segment_reshaped)
  # step1_output = tf.cast(tf.argmax(tf.math.reduce_mean(model_step1(embeddings), axis=0), axis=0), tf.float32)
  step2_output = tf.math.reduce_mean(model_step2(embeddings), axis=0)
  final_output = tf.nn.softmax(step2_output, name='final_output')
  
  serving_model = tf.keras.Model(input_segment, final_output, name='bruit_model_onestep')
  serving_model.save(ONESTEP_MODELDIR, include_optimizer=False)
  
  converter = tf.lite.TFLiteConverter.from_saved_model(ONESTEP_MODELDIR)
  
  tflite_model = converter.convert()
  
  with open(os.path.join(ONESTEP_MODELDIR, 'bruit_yamnet_onestep.tflite'), 'wb') as f:
    f.write(tflite_model)
