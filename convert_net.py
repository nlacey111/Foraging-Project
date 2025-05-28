import torch
import torch.nn as nn

# network_copy_args = dict(obs_space = env.observation_space, action_space =env.action_space, net_arch = net_arch, activation_fn = activation_fn)


class CustomNetwork(nn.Module):
    def __init__(self, network_args = None ):
        # Make network
        super(CustomNetwork, self).__init__()

        self.net_arch = network_args['net_arch']
        self.activation_fn = network_args['activation_fn'] 
        
        self.obs_space = network_args['obs_space']
        self.action_space = network_args['action_space']

        self.mod_list = self.make_modlist()

        self.net = nn.Sequential(
            nn.Linear(self.obs_space, self.net_arch[0]),
            nn.ReLU(),
            *self.mod_list,  # Unpack the module list
            nn.Linear(self.net_arch[-1], self.action_space),  # Assuming the last layer of mod_list is connected to this
        )

    def forward(self, x):
        # Process and give me activations
        activations = []

        for i,l in enumerate(self.net):
            x = self.net[i//2](x) + l(x)
            activations.append(x)

        return activations
    
    def make_modlist(self):
        module_list = []
        for i in range(len(self.net_arch)-1):
            module_list.append(nn.Linear(self.net_arch[i], self.net_arch[i+1]))
            module_list.append(nn.ReLU())
        return module_list

    # net1 is original, net2 is the one to copy to  
    def copy_weights(net1, net2):
        params1 = list(net1.get_parameters())
        params2 = list(net2.get_parameters())
        for i in range(len(params1)):
            params2[i].set(params1[i])
        return net2
    

    

