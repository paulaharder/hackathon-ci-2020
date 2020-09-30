import numpy as np
import tensorflow as tf
import os
import time

#_____________________
#Loading and preprocessing the data
#_____________________

my_path = '/home/harder/imagery'
CloudTop = np.load(my_path + '/X_train_CI20.npy')
TrueColor = np.load(my_path + '/Y_train_CI20.npy')

#sort out dark pictures (just quick naive method)
TrueColorNZ = np.delete(TrueColor,np.where(np.sum(TrueColor,axis=(1,2,3))<200000),0)
CloudTopNZ = np.delete(CloudTop,np.where(np.sum(TrueColor,axis=(1,2,3))<200000),0)

del TrueColor, CloudTop

IMG_WIDTH = 256
IMG_HEIGHT = 256

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image

train_dataset = []
for i in range(CloudTopNZ.shape[0]):
    ct = tf.cast(CloudTopNZ[i,:,:,:], tf.float32)
    tc = tf.cast(TrueColorNZ[i,:,:,:], tf.float32)
    ct, tc = resize(ct, tc,IMG_HEIGHT, IMG_WIDTH)
    ct, tc = normalize(ct, tc)
    ct = tf.expand_dims(ct, 0)
    tc = tf.expand_dims(tc, 0)
    train_dataset.append((ct, tc))
    
del TrueColorNZ, CloudTopNZ  

#_______________________
#Defining the model (source: tensorflow.org/tutorials/generative/pix2pix)
#_______________________

BUFFER_SIZE = 400
BATCH_SIZE = 1
EPOCHS = 62
    
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

OUTPUT_CHANNELS = 3

def Generator():
    inputs = tf.keras.layers.Input(shape=[256,256,3])
    down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]
    up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)
    x = inputs
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
  # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()

LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # ssim loss
    ssim_loss = tf.reduce_mean(1 - tf.image.ssim(target+1, gen_output+1,2.0))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, ssim_loss

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)
    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = my_path + '/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    generator_gradients = gen_tape.gradient(gen_total_loss,generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))

def fit(train_ds, epochs):
    for epoch in range(epochs):
        start = time.time()
        print("Epoch: ", epoch)
        for n  in range(len(train_ds)):
            (input_image, target) = train_ds[n]
            train_step(input_image, target, epoch)
        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
        checkpoint.save(file_prefix = checkpoint_prefix)

#____________
# Training the model
#____________

fit(train_dataset, EPOCHS)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

X_test = np.load(my_path + '/X_test_CI20_phase2.npy')

test_dataset = []
predictions = np.zeros(X_test.shape)
for i in range(X_test.shape[0]):
    ct = tf.cast(X_test[i,:,:,:], tf.float32)
    ct, tc = resize(ct, ct,IMG_HEIGHT, IMG_WIDTH)
    ct, tc = normalize(ct, tc)
    ct = tf.expand_dims(ct, 0)
    tc = tf.expand_dims(tc, 0)
    test_dataset.append((ct, tc))

for i in range(X_test.shape[0]):
    (test_input,_) = test_dataset[i]
    pred = generator(test_input,training=False)
    pred_crop = tf.squeeze(pred,0)
    pred_un = (pred_crop+1)*127.5
    pred_resize = tf.image.resize(pred_un, [127, 127],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    predictions[i,:,:,:] = pred_resize.numpy()
    
np.save(my_path + 'Y_test_CI20_phase2.npy', predictions)