from GAN_architecture import *
from config import *


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

if __name__ == "__main__":
    train_GAN(generator,
              discriminator,
              vgg,
              adversarial_model,
              data, 
              epochs = EPOCHS,
              batch_size = BATCH_SIZE,
              summary_writer = summary_writer)