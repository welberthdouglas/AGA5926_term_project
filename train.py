from GAN_architecture import *
from config import *
from utils import *

### TRAINING FUNCTION

def train_GAN(generator,
              discriminator,
              vgg,
              adversarial_model,
              data, 
              epochs,
              batch_size,
              summary_writer):
    
    batches_per_epoch = int(data[0].shape[0]/batch_size)
    step = 0
    for epoch in range(epochs):
        print(f"Epoch:{epoch}")
        for i in tqdm(range(batches_per_epoch)):
    
            """
            Train the discriminator network
            """
            # get a batch of images
            splus_images = data[0][i * batch_size:(i + 1) * batch_size]
            legacy_images = data[1][i * batch_size:(i + 1) * batch_size]
        
            # Generate high-resolution images from low-resolution images
            generated_images = generator.predict(splus_images)
        
            # Generate batch of real and fake labels
            SR_labels = np.ones((BATCH_SIZE, 16, 16, 1))
            LR_labels = np.zeros((BATCH_SIZE, 16, 16, 1))
        
            # Train the discriminator network on LR and SR images
            d_loss_real = discriminator.train_on_batch(legacy_images, SR_labels)
            d_loss_fake = discriminator.train_on_batch(generated_images, LR_labels)
        
            # Calculate total discriminator loss
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
            """
            Train the generator network
            """
        
            # Extract feature maps for real high-resolution images
            image_features = vgg.predict(legacy_images)
        
            # Train the generator network
            g_loss = adversarial_model.train_on_batch([splus_images, legacy_images],
                                             [SR_labels, image_features])
        
            # Write the losses to Tensorboard
            with summary_writer.as_default():
                tf.summary.scalar('generator_loss', g_loss[0], step=step)
                tf.summary.scalar('discriminator_loss', d_loss[0], step=step)
                summary_writer.flush()
                
            step +=1
        
        # Sample and save validation images after every 10 epochs
        if epoch % 10 == 0:
            
            with open('validation_data.pkl','rb') as f:
                 splus_val_images,legacy_val_images = pickle.load(f)
            
            #
            r_index = np.random.choice(range(25), 3, replace = False)
            
            # Asinh Shrink and Normalize images
            low_resolution_images = np.stack([splus_val_images[r_index[0]],
                                              splus_val_images[r_index[1]],
                                              splus_val_images[r_index[2]]])
            high_resolution_images = np.stack([legacy_val_images[r_index[0]],
                                               legacy_val_images[r_index[1]],
                                               legacy_val_images[r_index[2]]])
    
            generated_images = generator.predict_on_batch(low_resolution_images)
            generated_images = normalize(generated_images)
    
            for index, img in enumerate(generated_images):
                save_images(low_resolution_images[index], high_resolution_images[index], img,\
                            path="results/img_{}_{}".format(epoch, index))
        
        # Save model checkpoint after 100 epochs
        if epoch % 100 == 0:
            generator.save_weights("generator.h5")
            discriminator.save_weights("discriminator.h5")
            
    # Save final models
    generator.save_weights("generator.h5")
    discriminator.save_weights("discriminator.h5")

if __name__ == "__main__":
    
    vgg = build_vgg()
    discriminator = build_discriminator()
    generator = build_generator()
    adversarial_model = build_adversarial_model(generator,discriminator,vgg)
    
    try:
        with open('data.pkl','rb') as f:
             data = pickle.load(f)
    except:
        print("data.pkl not found, trying to preprocess data from raw fits ...")
        data = get_data()
        
    dir_writer = LOG_DIR + "train/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(dir_writer)
    
    train_GAN(generator,
              discriminator,
              vgg,
              adversarial_model,
              data, 
              epochs = EPOCHS,
              batch_size = BATCH_SIZE,
              summary_writer = summary_writer)
