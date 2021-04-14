## TO DO add a parser

"""
Imports and Global experimental configuration parameters
"""


class Arguments():
    def __init__(self):
        self.pgd = True
        self.detector = 6  # 2 means norm bound, 1 means krum and 0 nothing, 3 is GAN-based, 4 means opt_armor
        self.attack = 2  # 1 means random sampling, 3 means single shots 5 means no attack, 4 means model replacement
        self.single_shot_round = 30
        self.attackers_list = [0]
        self.enable_detector = False
        self.attack_step = 1
        self.attakType = 1  # 1 means data poisoning
        self.start_round = 0
        self.eps = 0.1  # parameter of pgd
        self.diffPrivacy = True
        self.bound = 5
        self.log_name = "newGANMNIST_updated_.csv"
        self.epsilon = 8
        self.delta = 10 ** (-5)
        self.n_batch = 3
        self.N_LOTS = 3
        self.batch_size = 64
        self.gan_loss_th1 = 0.11
        self.gan_loss_th2 = 0.11
        self.opt_loss_th = 100
        self.latent_size = 64
        self.hidden_size = 256
        self.image_size = 784
        self.max_iterations_opt = 300
        self.max_loss_opt = 2
        self.max_retries = 2
        self.ignore_detection_th = 10
        self.model_to_load = "mnist_model.pt"  # "model_attack_.pt"#"./model_35.pt"
        self.savemodel = 10
        ###############################

        self.test_batch_size = 64
        self.beta1 = 0.5
        self.no_cuda = False
        self.seed = 0
        self.log_interval = 10
        self.outf = '.'
        self.save_model = True

        #############################
        self.gpu = True
        self.num_users = 10
        self.iid = False
        self.unequal = True
        self.dataset = 'mnist'
        self.model = 'cnn'
        self.num_channels = 1
        self.num_classes = 10
        self.epochs = 110
        self.frac = 1
        self.local_bs = 10
        self.local_ep = 2
        self.lr = 0.01
        self.momentum = 0.5
        self.optimizer = 'sgd'
        self.verbose = 1
        self.attacker_index = 0
        #####NEW
        self.ganepochs = 100
        self.ganBatchSize = 128
        self.g_step = 1
        self.d_step = 1
