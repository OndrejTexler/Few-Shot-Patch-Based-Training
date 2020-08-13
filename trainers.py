import time
import models
import numpy as np
import six
import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from custom_transforms import *
from data import DatasetFullImages
import os


class Trainer(object):
    def __init__(self,
                 train_loader, data_for_dataloader, opt_discriminator, opt_generator,
                 reconstruction_criterion, adversarial_criterion, reconstruction_weight,
                 adversarial_weight, log_interval, scalar_logger, model_logger,
                 perception_loss_model,  perception_loss_weight, use_image_loss, device
                 ):

        self.train_loader = train_loader
        self.data_for_dataloader = data_for_dataloader

        self.opt_discriminator = opt_discriminator
        self.opt_generator = opt_generator

        self.reconstruction_criterion = reconstruction_criterion
        self.adversarial_criterion = adversarial_criterion

        self.reconstruction_weight = reconstruction_weight
        self.adversarial_weight = adversarial_weight

        self.scalar_logger = scalar_logger
        self.model_logger = model_logger

        self.training_log = {}
        self.log_interval = log_interval

        self.perception_loss_weight = perception_loss_weight
        self.perception_loss_model = perception_loss_model

        self.use_adversarial_loss = False
        self.use_image_loss = use_image_loss
        self.device = device

        self.dataset = None
        self.imloader = None


    def run_discriminator(self, discriminator, images):
        return discriminator(images)

    def compute_discriminator_loss(self, generator, discriminator, batch):
        generated = generator(batch['pre'])
        fake = self.apply_mask(generated, batch, 'pre_mask')
        fake_labels, _ = self.run_discriminator(discriminator, fake.detach())

        true = self.apply_mask(batch['already'], batch, 'already_mask')
        true_labels, _ = self.run_discriminator(discriminator, true)

        discriminator_loss = self.adversarial_criterion(fake_labels, self.zeros_like(fake_labels)) + \
                             self.adversarial_criterion(true_labels, self.ones_like(true_labels))

        return discriminator_loss

    def compute_generator_loss(self, generator, discriminator, batch, use_gan, use_mask):
        image_loss = 0
        perception_loss = 0
        adversarial_loss = 0

        generated = generator(batch['pre'])

        if use_mask:
            generated = generated * batch['mask']
            batch['post'] = batch['post'] * batch['mask']

        if self.use_image_loss:
            if generated[0][0].shape != batch['post'][0][0].shape:
                if ((batch['post'][0][0].shape[0] - generated[0][0].shape[0]) % 2) != 0:
                    raise RuntimeError("batch['post'][0][0].shape[0] - generated[0][0].shape[0] must be even number")
                if generated[0][0].shape[0] != generated[0][0].shape[1] or batch['post'][0][0].shape[0] != batch['post'][0][0].shape[1]:
                    raise RuntimeError("And also it is expected to be exact square ... fix it if you want")
                boundary_size = int((batch['post'][0][0].shape[0] - generated[0][0].shape[0]) / 2)
                cropped_batch_post = batch['post'][:, :, boundary_size: -1*boundary_size, boundary_size: -1*boundary_size]
                image_loss = self.reconstruction_criterion(generated, cropped_batch_post)
            else:
                image_loss = self.reconstruction_criterion(generated, batch['post'])

        if self.perception_loss_model is not None:
            _, fake_features = self.perception_loss_model(generated)
            _, target_features = self.perception_loss_model(Variable(batch['post'], requires_grad=False))
            perception_loss = ((fake_features - target_features) ** 2).mean()


        if self.use_adversarial_loss and use_gan:
            fake = self.apply_mask(generated, batch, 'pre_mask')
            fake_smiling_labels, _ = self.run_discriminator(discriminator, fake)
            adversarial_loss = self.adversarial_criterion(fake_smiling_labels, self.ones_like(fake_smiling_labels))

        return image_loss, perception_loss, adversarial_loss, generated


    def train(self, generator, discriminator, epochs, data_root, config_yaml_name, starting_batch_num):
        self.use_adversarial_loss = discriminator is not None
        batch_num = starting_batch_num
        save_num = 0

        start = time.time()
        for epoch in range(epochs):
            np.random.seed()
            for i, batch in enumerate(self.train_loader):
                # just sets the models into training mode (enable BN and DO)
                [m.train() for m in [generator, discriminator] if m is not None]
                batch = {k: batch[k].to(self.device) if isinstance(batch[k], torch.Tensor) else batch[k]
                         for k in batch.keys()}

                # train discriminator
                if self.use_adversarial_loss:
                    self.opt_discriminator.zero_grad()
                    discriminator_loss = self.compute_discriminator_loss(generator, discriminator, batch)
                    discriminator_loss.backward()
                    self.opt_discriminator.step()

                # train generator
                self.opt_generator.zero_grad()

                g_image_loss, g_perc_loss, g_adv_loss, _ = self.compute_generator_loss(generator, discriminator, batch, use_gan=True, use_mask=False)

                generator_loss = self.reconstruction_weight * g_image_loss + \
                                 self.perception_loss_weight * g_perc_loss + \
                                 self.adversarial_weight * g_adv_loss

                generator_loss.backward()

                self.opt_generator.step()

                # log losses
                current_log = {key: value.item() for key, value in six.iteritems(locals()) if
                               'loss' in key and isinstance(value, Variable)}

                self.add_log(current_log)

                batch_num += 1

                if batch_num % 100 == 0:
                    print(f"Batch num: {batch_num}, totally elapsed {(time.time() - start)}", flush=True)

                #if batch_num % self.log_interval == 0 or batch_num == 1:
                if batch_num % self.log_interval == 0 or batch_num == 1: #  (time.time() - start) > 16:
                    eval_start = time.time()
                    generator.eval()
                    self.test_on_full_image(generator, batch_num, data_root, config_yaml_name)
                    self.flush_scalar_log(batch_num, time.time() - start)
                    self.model_logger.save(generator, save_num, True)
                    #self.model_logger.save(discriminator, save_num, False)
                    save_num += 1
                    print(f"Eval of batch: {batch_num} took {(time.time() - eval_start)}", flush=True)

                    #if batch_num > 5000:
                    #    sys.exit(0)

        self.model_logger.save(generator, 99999)

    # Accumulates the losses
    def add_log(self, log):
        for k, v in log.items():
            if k in self.training_log:
                self.training_log[k] += v
            else:
                self.training_log[k] = v

    # Divide the losses by log_interval and print'em
    def flush_scalar_log(self, batch_num, took):
        for key in self.training_log.keys():
            self.scalar_logger.scalar_summary(key, self.training_log[key] / self.log_interval, batch_num)

        log = "[%d]" % batch_num
        for key in sorted(self.training_log.keys()):
            log += " [%s] % 7.4f" % (key, self.training_log[key] / self.log_interval)

        log += ". Took {}".format(took)
        print(log, flush=True)
        self.training_log = {}

    # Test the intermediate model on data from _gen folder
    def test_on_full_image(self, generator, batch_num, data_root, config_yaml_name):
        config_yaml_name = config_yaml_name.replace("reference", "").replace(".yaml", "")

        data_root = data_root.replace("_train", "_gen")
        if self.dataset is None:
            self.dataset = DatasetFullImages(data_root + "/" + self.data_for_dataloader['dir_pre'].split("/")[-1],
                                    "ignore",  # data_root + "/" + "ebsynth",
                                    "ignore",  # data_root + "/" + "mask",
                                    self.device,
                                    dir_x1=data_root + "/" + self.data_for_dataloader['dir_x1'].split("/")[-1] if self.data_for_dataloader['dir_x1'] is not None else None,
                                    dir_x2=data_root + "/" + self.data_for_dataloader['dir_x2'].split("/")[-1] if self.data_for_dataloader['dir_x2'] is not None else None,
                                    dir_x3=data_root + "/" + self.data_for_dataloader['dir_x3'].split("/")[-1] if self.data_for_dataloader['dir_x3'] is not None else None,
                                    dir_x4=data_root + "/" + self.data_for_dataloader['dir_x4'].split("/")[-1] if self.data_for_dataloader['dir_x4'] is not None else None,
                                    dir_x5=data_root + "/" + self.data_for_dataloader['dir_x5'].split("/")[-1] if self.data_for_dataloader['dir_x5'] is not None else None,
                                    dir_x6=data_root + "/" + self.data_for_dataloader['dir_x6'].split("/")[-1] if self.data_for_dataloader['dir_x6'] is not None else None,
                                    dir_x7=data_root + "/" + self.data_for_dataloader['dir_x7'].split("/")[-1] if self.data_for_dataloader['dir_x7'] is not None else None,
                                    dir_x8=data_root + "/" + self.data_for_dataloader['dir_x8'].split("/")[-1] if self.data_for_dataloader['dir_x8'] is not None else None,
                                    dir_x9=data_root + "/" + self.data_for_dataloader['dir_x9'].split("/")[-1] if self.data_for_dataloader['dir_x9'] is not None else None)
            self.imloader = torch.utils.data.DataLoader(self.dataset, 1, shuffle=False, num_workers=1, drop_last=False)  # num_workers=4

        with torch.no_grad():
            log = "### \n"
            log = log + "[%d]" % batch_num + " "
            generator_loss_on_ebsynth = 0
            for i, batch in enumerate(self.imloader):
                batch = {k: batch[k].to(self.device) if isinstance(batch[k], torch.Tensor) else batch[k]
                         for k in batch.keys()}
                g_image_loss, g_perc_loss, g_adv_loss, e_cls_loss, e_smiling_loss, gan_output =\
                        0, 0, 0, 0, 0, generator(batch['pre'])

                generator_loss = self.reconstruction_weight * g_image_loss + \
                                 self.perception_loss_weight * g_perc_loss + \
                                 self.adversarial_weight * g_adv_loss

                if True or batch['file_name'][0] != "111.png":  # do not accumulate loss in train frame
                    generator_loss_on_ebsynth = generator_loss_on_ebsynth + generator_loss

                if True or batch['file_name'][0] in ["111.png", "101.png", "106.png", "116.png", "121.png"]:
                    #log = log + batch['file_name'][0]
                    #log = log + ": %7.4f" % generator_loss + ", "

                    image_space = to_image_space(gan_output.cpu().data.numpy())

                    gt_test_ganoutput_path = data_root + "/" + "res_" + config_yaml_name
                    if not os.path.exists(gt_test_ganoutput_path):
                        os.mkdir(gt_test_ganoutput_path)
                    gt_test_ganoutput_path_batch_num = gt_test_ganoutput_path + "/" + str("%07d" % batch_num)
                    if not os.path.exists(gt_test_ganoutput_path_batch_num):
                        os.mkdir(gt_test_ganoutput_path_batch_num)
                    for k in range(0, len(image_space)):
                        im = image_space[k].transpose(1, 2, 0)
                        Image.fromarray(im).save(os.path.join(gt_test_ganoutput_path_batch_num, batch['file_name'][k]))
                        if i == 0:
                            Image.fromarray(im).save(os.path.join(gt_test_ganoutput_path, str("%07d" % batch_num) + ".png"))

            log = log + " totalLossOnEbsynth: %7.4f" % (generator_loss_on_ebsynth/(len(self.imloader)))
            print(log, flush=True)


    def apply_mask(self, x, batch, mask_key):
        if mask_key in batch:
            mask = Variable(batch[mask_key].expand(x.size()), requires_grad=False)
            return x * (mask / 2 + 0.5)
        return x

    def ones_like(self, x):
        return torch.ones_like(x).to(self.device)

    def zeros_like(self, x):
        return torch.zeros_like(x).to(self.device)

    @staticmethod
    def to_image_space(x):
        return ((np.clip(x, -1, 1) + 1) / 2 * 255).astype(np.uint8)
