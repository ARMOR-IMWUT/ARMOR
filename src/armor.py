""" The GAN based detector
@ Rania : I added the detection threshold 
"""

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        




def test(model, fake, target_label):
        #compute the accuracy of the fake image on the classifier
        output = model(fake)
        output = torch.exp(output)
        out = torch.narrow(output, 1, target_label, 1)
        out = torch.squeeze(out)
        pred = np.mean(output.tolist(), axis = 0)
        D_G_z2 = out.mean().item()
        print("Predictions: ", pred)
        print("Argmax: ", np.argmax(pred), "accuracy: ", D_G_z2)
        
class Discriminator(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)
        self.args = args

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
class Generator_28(nn.Module):
    def __init__(self, nz=64):
        super(Generator_28, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.Linear(self.nz, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )
    def forward(self, x):
        x  = x.view(-1, 64,64)
        return self.main(x).view(-1, 1, 28, 28)

"""class Generator(nn.Module):
    def __init__(self, d=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(100, d*4, 4, bias=False),
            nn.BatchNorm2d(d*4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(d*4, d*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(d*2, d, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(d, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)"""

"""class Generator(nn.Module):
    def __init__(self, nz=100):
        super(Generator, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.Linear(self.nz, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)"""



class Generator(nn.Module):
    def __init__(self, latent_size, hidden_size, image_size):
         super(Generator, self).__init__()
         # 1
         self.fc1 = nn.Linear(latent_size, hidden_size)
         self.fc2 = nn.Linear(hidden_size, hidden_size)
         self.fc3 = nn.Linear(hidden_size, image_size)

    
    def forward(self, x):
         # 4
         x = F.relu(self.fc1(x)) # (input, negative_slope=0.2)
         x = F.relu(self.fc2(x))
         out = F.tanh(self.fc3(x))
         #print(out.shape)
         return out



def test(args, model, device, test_loader, target_label):
    model.eval()
    test_loss = 0
    correct = 0
    output_average  = torch.zeros([10], dtype = torch.float).to(device)
    criterion = nn.NLLLoss()
    with torch.no_grad():
        
        data, target = test_loader.to(device), torch.tensor([target_label] *test_loader.size(0))
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item() # sum up batch loss
        pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
        correct += pred.eq(target.view_as(pred)).sum().item()
        print(target)
        print(pred)
    output_average  = torch.mean(output,dim = 0) 
    test_loss /= test_loader.size()[0]
    accuracy = 100. * correct / test_loader.size()[0]
    #print("Predictions *** : ", output_average)
    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    test_loss, correct, test_loader.size()[0],
    #    100. * correct / test_loader.size()[0]))
    
    return test_loss, accuracy, output_average


def model_replacement(attack_model, global_model, num_users, args):
    
    
    list_global_model = orderdict_tolist(global_model)
    list_attack_model = orderdict_tolist(attack_model)
    results = list(map(add, list_global_model,  np.subtract(list_attack_model,list_global_model)*num_users))
    
    return list_todict(results, args)


class  GAN_ARMOR (object) :

    def __init__(self, args):
      
      self.discriminators = []
      self.generators = []
      self.optimizersG = []
      self.optimizersD = []
      self.args = args 
      self.ratio1Hist = []
      self.ratio2Hist = []
      self.device = 'cuda' if args.gpu else 'cpu'
      self.criterion = nn.NLLLoss().to(self.device)
      
      for label in range (0, args.num_classes):
        if(args.dataset=="mnist"):
            discriminator =  CNNMnist(args,True,label)
        else:
            discriminator =  CNNFashion_Mnist(args,True,label)
        
        discriminator.apply(weights_init)
        discriminator = discriminator.to(device)
        optimizerD = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(args.beta1, 0.999))
        self.discriminators.append(discriminator)
        self.optimizersD.append(optimizerD)
        netG = Generator_28().to(device)
        netG.apply(weights_init)
        optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(args.beta1, 0.999))
        self.generators.append(netG)
        self.optimizersG.append(optimizerG)

    def train_generator(self, args, label, epoch):
       

        if(args.dataset=="mnist"):
            discriminator =  CNNMnist(args,True,label)
        else:
            discriminator =  CNNFashion_Mnist(args,True,label)
            
        discriminator.apply(weights_init)
        discriminator = discriminator.to(device)
        netD = discriminator
        netG = Generator_28().to(device)
        netG.apply(weights_init)
        
        optimizerD = optim.Adam(netD.parameters(), lr=0.00000001, betas=(args.beta1, 0.999))
        #optimizerD = optim.Adam(netD.parameters(), lr=0.00001, betas=(args.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(args.beta1, 0.999))
        criterion =  nn.NLLLoss().to(self.device)
        gradients = []


        #I kept this code in this way to adapt it later to the case where num_users>epochs 
        """for gradient in self.weights:
            gradients.append(torch.Tensor(orderdict_tolist_adapt(gradient)))"""
        
        
        
        """state = netD.state_dict()
        if(args.dataset=="mnist"):
            state["fc2.weight"][10] = state["fc2.weight"][10] * 2
            state["fc2.bias"][10] = state["fc2.bias"][10] * 2   
        else:
            state["fc1.weight"][10] = state["fc1.weight"][10] * 2
            state["fc1.bias"][10] = state["fc1.bias"][10] * 2   """      
            
        
        #netD_list = torch.Tensor(orderdict_tolist(state))

        diff_model = (1/(args.ganepochs))* (torch.Tensor(orderdict_tolist_adapt(self.new_model.state_dict(), args.dataset=="fmnist")))#- torch.Tensor(orderdict_tolist_adapt(self.old_model.state_dict(), args.dataset=="fmnist")))
        
        """gradients.append(diff_model)
        gradients.append(netD_list)
        n = len(gradients)       
        avg_grad = sum(grad for grad in gradients).div_(n)
        netD.load_state_dict(list_todict(avg_grad, args, True, label))"""
        
        #netD.load_state_dict(list_todict(diff_model, args, True, label))
        
        
        for epch in range(args.ganepochs):
            #train the descriminator using real data if we still have some local grads         
            
            if epch % args.d_step == 0:
                
                
              
              netD.zero_grad()
              z = torch.randn(args.batch_size, args.latent_size).cuda()
              z = Variable(z)
              fake = netG(z)
              labelD = torch.full((args.batch_size,), args.num_classes, device=self.device, dtype = torch.long)
              outD = netD(fake)
              #print("the prediction for desc: ", outD.argmax(1, keepdim=True))
            
              #outD = torch.exp(output)
              #isolate the probabilities of the target label
              #outD = torch.narrow(output, 1, label, 1)
              #outD = torch.squeeze(outD)
              errD = criterion(outD, labelD)
              #print("errD: ", errD)
              errD.backward()
              
              state = netD.state_dict()


              """ if(args.dataset=="mnist"):
                    state["fc2.weight"][10] = state["fc2.weight"][10] * 2
                    state["fc2.bias"][10] = state["fc2.bias"][10] * 2   
              else:
                    state["fc1.weight"][10] = state["fc1.weight"][10] * 2
                    state["fc1.bias"][10] = state["fc1.bias"][10] * 2    """   
              
              netD_list = torch.Tensor(orderdict_tolist(state))  
              
              tmp = sum(grad for grad in [2*netD_list,diff_model]).div(2)     
              netD.load_state_dict(list_todict(tmp, args, True, label))
            
              optimizerD.step()
            
            
            

            #train the generator
            
            
            if epch % args.g_step == 0:
                netG.zero_grad()
                z = torch.randn(args.batch_size, args.latent_size).cuda()
                z = Variable(z)
                fake = netG(z)
                labelG = torch.full((args.batch_size,),label, device=self.device, dtype = torch.long)
                outG = netD(fake)
                #print("the prediction of desc: ", outG.argmax(1, keepdim=True))
                #outG = torch.exp(output)
                #isolate the probabilities of the target label
                #outG = torch.narrow(output, 1, label, 1)
                #outG = torch.squeeze(outG)
                errG = criterion(outG, labelG)
                #print("errG: ", errG)
                errG.backward()
                optimizerG.step()
                out_fake = fake.detach()
                for i in range(100):

                    z = torch.randn(args.batch_size, args.latent_size).cuda()
                    z = Variable(z)
                    fake = netG(z)
                    out_fake =  torch.cat((out_fake, fake.detach() ), 0)
            loss_inspected_, accuracy_inspected_ , average_output_inspected_= test(args, self.new_model, device, out_fake, label)
            if (accuracy_inspected_ > 70):
                break
          
            


        #self.discriminators[label] = netD 
        #self.generators[label] = netG
        #self.optimizersD[label] = optimizerD
        #self.optimizersG[label] = optimizerG 


        #compute the accuracy of the target_label
        D_G_z2 = outG.mean().item()
        D_D_z2 = outD.mean().item()
        pred = np.mean(outG.tolist(), axis = 0)
        print("Predictions: ", pred)
        print("Argmax: ", np.argmax(pred), "accuracy: ", D_G_z2)
        
        vutils.save_image(fake.data[:64],
                          './fake_sample_'+args.dataset+'_label' + str(label) +'_epoch'+str(epoch)+ '.png',
                              normalize=True)
        
        
        return out_fake
             

    def train_generator_modif(self, args, label, epoch):
       
        if(args.dataset=="mnist"):
            discriminator =  CNNMnist(args,True,label)
        else:
            discriminator =  CNNFashion_Mnist(args,True,label)
            
        discriminator.apply(weights_init)
        discriminator = discriminator.to(device)
        netD = discriminator
        netG = Generator_28().to(device)
        netG.apply(weights_init)
        
        # optimizerD = optim.Adam(netD.parameters(), lr=0.00000001, betas=(args.beta1, 0.999))
        optimizerD = optim.LBFGS(netD.parameters(), lr=0.0001)
        optimizerG = optim.LBFGS(netG.parameters(), lr=0.01)
        criterion =  nn.NLLLoss().to(self.device)
        gradients = []


        #I kept this code in this way to adapt it later to the case where num_users>epochs 
        """for gradient in self.weights:
            gradients.append(torch.Tensor(orderdict_tolist_adapt(gradient)))"""
        
        
        
        """state = netD.state_dict()
        if(args.dataset=="mnist"):
            state["fc2.weight"][10] = state["fc2.weight"][10] * 2
            state["fc2.bias"][10] = state["fc2.bias"][10] * 2   
        else:
            state["fc1.weight"][10] = state["fc1.weight"][10] * 2
            state["fc1.bias"][10] = state["fc1.bias"][10] * 2   """      
            
        
        #netD_list = torch.Tensor(orderdict_tolist(state))

        diff_model = (1/(args.ganepochs))* (torch.Tensor(orderdict_tolist_adapt(self.new_model.state_dict(), args.dataset=="fmnist"))- torch.Tensor(orderdict_tolist_adapt(self.old_model.state_dict(), args.dataset=="fmnist")))
        
        """gradients.append(diff_model)
        gradients.append(netD_list)
        n = len(gradients)       
        avg_grad = sum(grad for grad in gradients).div_(n)
        netD.load_state_dict(list_todict(avg_grad, args, True, label))"""
        
        #netD.load_state_dict(list_todict(diff_model, args, True, label))
        
        
        for epch in range(args.ganepochs):
            #train the descriminator using real data if we still have some local grads         

              def closure_disc():

                    netD.zero_grad()
                    z = torch.randn(args.batch_size, args.latent_size).cuda()
                    z = Variable(z)
                    fake = netG(z)
                    labelD = torch.full((args.batch_size,), args.num_classes, device=self.device, dtype = torch.long)
                    outD = netD(fake)
                    errD = criterion(outD, labelD)
                    errD.backward()
                    return errD


              def closure_gen():

                    netG.zero_grad()
                    z = torch.randn(args.batch_size, args.latent_size).cuda()
                    z = Variable(z)
                    fake = netG(z)
                    labelG = torch.full((args.batch_size,),label, device=self.device, dtype = torch.long)
                    outG = netD(fake)
                    #print("the prediction of desc: ", outG.argmax(1, keepdim=True))
                    #outG = torch.exp(output)
                    #isolate the probabilities of the target label
                    #outG = torch.narrow(output, 1, label, 1)
                    #outG = torch.squeeze(outG)
                    errG = criterion(outG, labelG)
                    #print("errG: ", errG)
                    errG.backward()
                    return errG
              
              state = netD.state_dict()

              """ if(args.dataset=="mnist"):
                    state["fc2.weight"][10] = state["fc2.weight"][10] * 2
                    state["fc2.bias"][10] = state["fc2.bias"][10] * 2   
              else:
                    state["fc1.weight"][10] = state["fc1.weight"][10] * 2
                    state["fc1.bias"][10] = state["fc1.bias"][10] * 2    """   
              
              netD_list = torch.Tensor(orderdict_tolist(state))  
              
              tmp = sum(grad for grad in [2*netD_list,diff_model]).div(2)     
              netD.load_state_dict(list_todict(tmp, args, True, label))
            
              optimizerD.step(closure_disc)
              optimizerG.step(closure_gen)
              z = torch.randn(args.batch_size, args.latent_size).cuda()
              z = Variable(z)
              fake = netG(z)
              out_fake = fake.detach()
              for i in range(100):

                    z = torch.randn(args.batch_size, args.latent_size).cuda()
                    z = Variable(z)
                    fake = netG(z)
                    out_fake =  torch.cat((out_fake, fake.detach() ), 0)
            
              loss_inspected_, accuracy_inspected_ , average_output_inspected_= test(args, self.new_model, device, out_fake, label)
              if (accuracy_inspected_ > 70):
                    break
          
            


        #self.discriminators[label] = netD 
        #self.generators[label] = netG
        #self.optimizersD[label] = optimizerD
        #self.optimizersG[label] = optimizerG 


        #compute the accuracy of the target_label
        D_G_z2 = outG.mean().item()
        D_D_z2 = outD.mean().item()
        pred = np.mean(outG.tolist(), axis = 0)
        print("Predictions: ", pred)
        print("Argmax: ", np.argmax(pred), "accuracy: ", D_G_z2)
        
        vutils.save_image(fake.data[:64],
                          './fake_sample_'+args.dataset+'_label' + str(label) +'_epoch'+str(epoch)+ '.png',
                              normalize=True)
        
        
        return out_fake
                
    def poisoningDetection(self, epoch): 
        args = self.args
        
        
         
        if (args.dataset == "fmnist"):
          model_poisoned = CNNFashion_Mnist(args)
          model_good_1 =  CNNFashion_Mnist(args)
          model_good_2 = CNNFashion_Mnist(args)
        else: 
          model_poisoned = CNNMnist(args)
          model_good_1 = CNNMnist(args)
          model_good_2 = CNNMnist(args)

        model_poisoned.load_state_dict(torch.load("model_current_.pt"))
        model_poisoned = model_poisoned.to(device)

       
        model_good_1.load_state_dict(torch.load("model_last_.pt"))
        model_good_1 = model_good_1.to(device)
        
        
       
        model_good_2.load_state_dict(torch.load("model_last2_.pt"))
        model_good_2 = model_good_2.to(device)
        

        
        self.new_model= model_poisoned
        self.old_model = model_good_1
        self.second_old_model = model_good_2

        import time
        diff_loss, loss_inspected, loss_clean, loss_clean2, diff_accuracy, accuracy_clean, accuracy_inspected, average_output_clean, average_output_clean2, average_output_inspected, ratio1_, ratio2_ = ([] for i in range(12))
        # starting time
        start = time.time()
        poisoning_flag = False 
      
        ratio1_list = []
        ratio2_list = []
        
        for label in range(0, args.num_classes):   
          print(self.ratio1Hist)
          print(self.ratio2Hist)
          if(epoch > args.ignore_detection_th) :
             
             tn = torch.tensor(self.ratio1Hist)
             means1 = torch.mean(tn, dim=0)
             tn = torch.tensor(self.ratio2Hist)
             means2 = torch.mean(tn, dim=0)
               
          
          print(f'------------------label: {label}--------------------')
          
          fake_poisoned = self.train_generator(args, label, epoch)
          
          loss_inspected_, accuracy_inspected_ , average_output_inspected_= test(args, model_poisoned, device, fake_poisoned, label)
          loss_clean_, accuracy_clean_ , average_output_clean_= test(args, model_good_1, device, fake_poisoned, label)
          loss_clean2_, accuracy_clean2_ , average_output_clean2_= test(args, model_good_2, device, fake_poisoned, label)
          
          average_output_inspected_ = torch.exp(average_output_inspected_)
          average_output_clean_ = torch.exp(average_output_clean_)
          average_output_clean2_ = torch.exp(average_output_clean2_)
        
          print('Label{}, epoch {}'.format(label, epoch))
          print("average prediction t+1: ", average_output_inspected_)
          print("average prediction t: ", average_output_clean_)
          print("average prediction t-1: ", average_output_clean2_)
          
            
          accuracy_inspected.append(accuracy_inspected_)
          accuracy_clean.append(accuracy_clean_)
          average_output_inspected.append(average_output_inspected_)
          
          loss_clean.append(loss_clean_)
          loss_clean2.append(loss_clean2_)
          loss_inspected.append(loss_inspected_)
          
         
          
          average_output_clean.append(average_output_clean_)
          average_output_clean2.append(average_output_clean2_)
          
          diff_loss_ = np.abs(loss_inspected_ - loss_clean_)
          diff_loss.append(diff_loss_)
          diff_accuracy_ = np.abs(accuracy_inspected_ - accuracy_clean_)
          diff_accuracy.append(diff_accuracy_)
          
          print("Loss t+1", loss_inspected_)
          print("Loss t", loss_clean_)
          print("Loss t-1", loss_clean2_)
          
          print("Accuracy t+1", accuracy_inspected_)
          print("Accuracy model t", accuracy_clean_)
          print("Accuracy model t-1", accuracy_clean2_)
          
          if (loss_inspected_ !=0 and loss_clean_ != 0 and loss_clean2_ != 0 ):
              ratio1 = np.abs(loss_inspected_ - loss_clean_) /  max(loss_inspected_,loss_clean_)#loss_clean_/ (loss_inspected_ + 10**(-15)) <
              ratio2 = np.abs(loss_inspected_ - loss_clean2_) /  max(loss_inspected_,loss_clean2_) #loss_clean2_/ (loss_inspected_ + 10**(-15))<
          
          else:
              ratio1 = 0
              ratio2 = 0
        
          print("Ratio 1 : ", ratio1)
          print("Ratio 2 : ", ratio2)
          ratio1_.append(ratio1)
          ratio2_.append(ratio2)
          if(epoch > args.ignore_detection_th-2) :
                
              ratio1_list.append(ratio1)
              ratio2_list.append(ratio2)  
            
          if(epoch > args.ignore_detection_th) :
              print("Mean1: ",  means1[label] )
              print("Mean2: ",  means2[label] )
             
              if ( ratio1 > args.gan_loss_th1 and  ratio2 > args.gan_loss_th2) :
                poisoning_flag = True 
                if (args.enable_detector):
                    break
        if (args.enable_detector):
            if (poisoning_flag ):
              model = model_good_1
            else:
              model = model_poisoned
              if(epoch > args.ignore_detection_th-2) :
                  self.ratio1Hist.append(ratio1_list)
                  self.ratio2Hist.append(ratio2_list)
        else: 
            model = model_poisoned
            if(epoch > args.ignore_detection_th-2) :
                  self.ratio1Hist.append(ratio1_list)
                  self.ratio2Hist.append(ratio2_list)
        # end time
        end = time.time()
        runtimeGAN = end - start

        # total time taken
        print(f"Runtime of the program is {runtimeGAN}")
        
       
        
        return poisoning_flag, runtimeGAN, model, loss_inspected, loss_clean,  loss_clean2, average_output_inspected,  average_output_clean, average_output_clean2, ratio1_, ratio2_



   
      
   
    


    
    


