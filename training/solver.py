import time
import datetime
import tensorflow.keras.backend as K
import os
from tqdm import tqdm
import tensorflow as tf
from data_loader.train_loader import TrainLoader
from data_loader.mtat_loader import DataLoader
from model import (
    TagEncoder,
    TagDecoder,
    WaveEncoder,
    WaveProjector,
    SupervisedClassifier,
)
from model import resemul
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# fix random seed
SEED = 42
tf.random.set_seed(SEED)


class Solver(object):
    def __init__(self, config):
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
        self.data_path = config.data_path
        self.model_save_path = config.model_save_path
        self.withJE = config.withJE
        self.data_path = config.data_path
        self.block = config.block
        self.stage1_test_template = "AELoss : {:.5f}"
        self.stage1_train_template = "Epoch: {}, TotalLoss : {:.4f},  AELoss : {:.4f}, PWLoss : {:.4f}"
        self.stage2_test_template = "Test Loss : {}, Test AUC : {:.4f}, Test PR-AUC : {:.4f}"
        self.stage2_train_template = "Epoch : {}, Loss : {:.4f}, AUC : {:.4f}, PR-AUC : {:.4f}"
        self.encoder_type = config.encoder_type
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.train_ds, self.valid_ds, self.test_ds = self.get_data(self.data_path)
        self.get_model()
        self.get_optimizer()
        self.get_loss()
        self.get_metric()
        self.start_time = time.time()


    def get_data(self, root="../../tf2-harmonic-cnn/dataset"):
        train_data = TrainLoader(root=root, split="train")
        valid_data = DataLoader(root=root, split="valid")
        test_data = DataLoader(root=root, split="test")
        return train_data, valid_data, test_data


    def get_model(self):
        if self.encoder_type == "HC":
            self.wave_encoder = WaveEncoder()
        elif self.encoder_type == "SC":
            self.wave_encoder = resemul(block_type=self.block)
        elif self.encoder_type == "MS":
            self.wave_encoder = resemul(ms=True, block_type="rese")
        
        self.wave_projector = WaveProjector(128)
        self.classifier = SupervisedClassifier()
        self.tag_encoder = TagEncoder()
        self.tag_decoder = TagDecoder(50)


    def get_optimizer(self):
        self.adam = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
        self.sgd2 = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)


    def get_loss(self):
        self.c_loss = tf.keras.losses.BinaryCrossentropy()
        self.ae_loss = tf.keras.losses.MeanSquaredError()
        self.pw_loss = tf.keras.losses.CosineSimilarity()


    def get_metric(self):
        self.s1_train_loss = tf.keras.metrics.Mean()
        self.s1_train_ae = tf.keras.metrics.Mean()
        self.s1_train_pw = tf.keras.metrics.Mean()
        self.s1_test_loss = tf.keras.metrics.Mean()
        self.s2_train_loss = tf.keras.metrics.Mean()
        self.s2_train_auc = tf.keras.metrics.AUC()
        self.s2_train_pr = tf.keras.metrics.AUC(curve='PR')
        self.s2_test_loss = tf.keras.metrics.Mean()
        self.s2_test_auc = tf.keras.metrics.AUC()
        self.s2_test_pr = tf.keras.metrics.AUC(curve='PR')


    @tf.function
    def stage1_adam_step(self, wave, labels):
        # apply GradientTape for differentiation
        with tf.GradientTape() as tape:
            # 1. prediction
            # wave
            # z1 shape : (1024,)
            # r1 shape : (128,)
            z1 = self.wave_encoder(wave, training=True)
            r1 = self.wave_projector(z1, training=True)

            # tag autoencoder
            # z2 shape : (128,)
            # predictions = (50,)
            z2 = self.tag_encoder(labels, training=True)
            predictions = self.tag_decoder(z2, training=True)

            # 2. calculate Loss
            # ae_loss : autoencoder loss
            # pw_loss : loss between rese and autoencoder
            ae_loss = self.ae_loss(labels, predictions) 
            pw_loss = (1-self.pw_loss(z2, r1)) + (1-self.pw_loss(1-z2, 1-r1))
            total_loss = ae_loss + pw_loss

        train_variable = (
            self.wave_encoder.trainable_variables
            + self.wave_projector.trainable_variables
            + self.tag_encoder.trainable_variables
            + self.tag_decoder.trainable_variables
        )

        # 3. calculate gradients
        gradients = tape.gradient(total_loss, train_variable)

        # 4. Backpropagation - update weight
        self.adam.apply_gradients(zip(gradients, train_variable))
        
        # update loss and accuracy
        self.s1_train_ae(ae_loss)
        self.s1_train_pw(pw_loss)
        self.s1_train_loss(total_loss)


    @tf.function
    def stage1_sgd_step(self, wave, labels):
        # apply GradientTape for differentiation
        with tf.GradientTape() as tape:
            # 1. prediction
            # wave
            # z1 shape : (1024,)
            # r1 shape : (128,)
            z1 = self.wave_encoder(wave, training=True)
            r1 = self.wave_projector(z1, training=True)

            # tag autoencoder
            # z2 shape : (128,)
            # predictions = (50,)
            z2 = self.tag_encoder(labels, training=True)
            predictions = self.tag_decoder(z2, training=True)

            # 2. calculate Loss
            # ae_loss : autoencoder loss
            # pw_loss : loss between rese and autoencoder
            ae_loss = self.ae_loss(labels, predictions)
            pw_loss = (1-self.pw_loss(z2, r1)) + (1-self.pw_loss(1-z2, 1-r1))
            total_loss = ae_loss + pw_loss

        train_variable = (
            self.wave_encoder.trainable_variables
            + self.wave_projector.trainable_variables
            + self.tag_encoder.trainable_variables
            + self.tag_decoder.trainable_variables
        )

        # 3. calculate gradients
        gradients = tape.gradient(total_loss, train_variable)

        # 4. Backpropagation - update weight
        self.sgd.apply_gradients(zip(gradients, train_variable))

        # update loss and accuracy
        self.s1_train_ae(ae_loss)
        self.s1_train_pw(pw_loss)
        self.s1_train_loss(total_loss)


    def stage1_train(self, epochs, optimizer):
        for epoch in range(epochs):
            for wave, labels in tqdm(self.train_ds):
                if optimizer == "adam":
                    self.stage1_adam_step(wave, labels)
                elif optimizer == "sgd":
                    self.stage1_sgd_step(wave, labels)

            stage1_log = self.stage1_train_template.format(
                epoch + 1, self.s1_train_ae.result(), self.s1_train_pw.result(), self.s1_train_loss.result()
            )
            print(stage1_log)


    @tf.function
    def stage1_test_step(self, wave, labels):
        z = self.tag_encoder(labels, training=False)
        predictions = self.tag_decoder(z, training=False)
        recon_loss = self.ae_loss(labels, predictions)
        self.s1_test_loss(recon_loss)


    @tf.function
    def stage2_adam_step(self, wave, labels):
        with tf.GradientTape() as tape:

            z = self.wave_encoder(wave, training=False)
            predictions = self.classifier(z, training=True)
            loss = self.c_loss(labels, predictions)

        train_variable = self.classifier.trainable_variables
        gradients = tape.gradient(loss, train_variable)
        
        self.adam.apply_gradients(zip(gradients, train_variable))

        self.s2_train_loss(loss)
        self.s2_train_auc(labels, predictions)
        self.s2_train_pr(labels, predictions)


    @tf.function
    def stage2_sgd_step(self, wave, labels):
        with tf.GradientTape() as tape:

            z = self.wave_encoder(wave, training=False)
            predictions = self.classifier(z, training=True)
            loss = self.c_loss(labels, predictions)

        train_variable = self.classifier.trainable_variables
        gradients = tape.gradient(loss, train_variable)

        self.sgd2.apply_gradients(zip(gradients, train_variable))

        self.s2_train_loss(loss)
        self.s2_train_auc(labels, predictions)
        self.s2_train_pr(labels, predictions)


    @tf.function
    def stage2_test_step(self, wave, labels):
        z = self.wave_encoder(wave, training=False)
        predictions = self.classifier(z, training=False)

        loss = self.c_loss(labels, predictions)
        self.s2_test_loss(loss)
        self.s2_test_auc(labels, predictions)
        self.s2_test_pr(labels, predictions)


    def stage2_train(self, epochs, optimizer):
        for epoch in range(epochs):
            for wave, labels in tqdm(self.train_ds):
                if optimizer == "adam":
                    self.stage2_adam_step(wave, labels)
                elif optimizer == "sgd":
                    self.stage2_sgd_step(wave, labels)

            stage2_log = self.stage2_train_template.format(
                epoch + 1, self.s2_train_loss.result(), self.s2_train_auc.result(), self.s2_train_pr.result()
            )
            print(stage2_log)

        if (epoch % 19 == 0 and epoch!= 0):
            for valid_wave, valid_labels in tqdm(self.valid_ds):
                self.stage2_test_step(valid_wave, valid_labels)

            valid_log = self.stage2_test_template.format(self.s2_test_loss.result(), self.s2_test_auc.result(), self.s2_test_pr.result())
            print(valid_log)


    def run_stage1(self):
        print("@@@@@@@@@@@@@@@@@@@Start training Stage 1@@@@@@@@@@@@@@@@@@\n")
        # training

        for i in range(3):
            if i == 0:
                epochs = 60
                self.stage1_train(epochs, optimizer="adam")
            elif i == 1:
                epochs = 20
                self.stage1_train(epochs, optimizer="sgd")
            else:
                epochs = 20
                new_lr = 0.0001
                self.sgd.lr.assign(new_lr)
                self.stage1_train(epochs, optimizer="sgd")

        for wave, labels in tqdm(self.test_ds):
            self.stage1_test_step(wave, labels)

        stage1_test_loss = self.stage1_test_template.format(self.s1_test_loss.result())
        print(stage1_test_loss)


    def run_stage2(self):
        print("\n\n@@@@@@@@@@@@@@@@@@@Start training Stage 2@@@@@@@@@@@@@@@@@@\n")
        for i in range(4):
            if i == 0:
                epochs = 60
                self.stage2_train(epochs, optimizer="adam")
            elif i == 1:
                epochs = 20
                self.stage2_train(epochs, optimizer="sgd")
            elif i ==2:
                epochs = 20
                new_lr = 0.0001
                self.sgd2.lr.assign(new_lr)
                self.stage2_train(epochs, optimizer="sgd")
            else:
                epochs= 100
                new_lr = 0.00001
                self.sgd2.lr.assign(new_lr)
                self.stage2_train(epochs, optimizer="sgd")

        for wave, labels in tqdm(self.test_ds):
            self.stage2_test_step(wave, labels)

        print("Time taken : ", time.time() - self.start_time)

        test_result = self.stage2_test_template.format(self.s2_test_loss.result(), self.s2_test_auc.result(), self.s2_test_pr.result())
        print(test_result)


    def run(self):
        self.run_stage1()
        self.run_stage2()
