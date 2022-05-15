import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

if(torch.cuda.is_available()):
    device = torch.device("cuda")
    print(device, torch.cuda.get_device_name(0))
else:
    device= torch.device("cpu")
    print(device)

N_STATES  = 243
N_ACTIONS = 39
HIDDEN_UNITS = 512

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(N_STATES, HIDDEN_UNITS),
            nn.LayerNorm(HIDDEN_UNITS),
            nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS),
            nn.LayerNorm(HIDDEN_UNITS),
            nn.Linear(HIDDEN_UNITS, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(N_STATES, HIDDEN_UNITS),
            nn.LayerNorm(HIDDEN_UNITS),
            nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS),
            nn.LayerNorm(HIDDEN_UNITS),
            nn.Linear(HIDDEN_UNITS, N_ACTIONS)
        )
        self.log_std = nn.Parameter(torch.ones(1, N_ACTIONS) * 0.0)
        self.apply(init_weights)
    
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value

def Interact_with_Unity_one_step (DecisionSteps):
    # ENV and NET are global variables        
    s = DecisionSteps.obs[0]  
    s = torch.FloatTensor(s)       
    dist, value = NET(s.to(device))
    a = dist.sample() 
    log_prob = dist.log_prob(a)
    
    a = a.cpu().detach().numpy()
    a = ActionTuple(np.array(a, dtype=np.float32))
    ENV.set_actions(BEHAVIOR_NAME, a)   
    ENV.step()
    a = a._continuous #convert from ActionTuple to np.array
    a = torch.FloatTensor(a) # convert from np.array to Tensor
    return s, value, a, log_prob

def Collect_REWARDS_and_MASKS (step, AgentSteps, flag): 
    #flag=1:decision, 0: terminal steps
    #REWARDS, MASKS, NEXT_STATES are gloable variables
    r = AgentSteps.reward
    r = torch.FloatTensor(r).unsqueeze(1)
    s = torch.FloatTensor(AgentSteps.obs[0])
    s = torch.FloatTensor(s).to(device) 
    for idx in range(len(AgentSteps)):
        AgentID = AgentSteps.agent_id[idx]
        REWARDS[step][AgentID]=r[idx]
        MASKS[step][AgentID]= flag
        NEXT_STATES[step][AgentID]=s[idx]

def collect_training_data (print_message):  
    # Interact with Unity INTERACTION_STEPS to collect training data
    # The total number of training data collected (buffer size) is INTERACTION_STEPS*N_AGENTS 
    #ENV, BEHAVIOR_NAME are gloabl variables
    #STATES, ACTIONS, LOG_PROBS, VALUES, REWARDS, MASKS, NEXT_STATES are global variables (tensor array)
    if(print_message):
        print("Collecting ", INTERACTION_STEPS, " training steps from ", N_AGENTS, " agents", end=": ")
    step = 0  
    DecisionSteps, TerminalSteps = ENV.get_steps(BEHAVIOR_NAME)
    while(step < INTERACTION_STEPS): #try to run TIME_HORIZON good steps
        #if we have no decision agents,then continue next loop without increase step
        if(len(DecisionSteps) == 0): 
            ENV.reset() 
            DecisionSteps, TerminalSteps = ENV.get_steps(BEHAVIOR_NAME)
            continue #continue next while loop without increase step
        
        # Interacts with Unity one step
        s, value, a, log_prob = Interact_with_Unity_one_step (DecisionSteps)
        NextDecisionSteps, NextTerminalSteps = ENV.get_steps(BEHAVIOR_NAME)

        #if this or next decision step misses some agents, then do not collect data
        if(len(DecisionSteps)!= N_AGENTS or len(NextDecisionSteps)!= N_AGENTS):
            DecisionSteps, TerminalSteps = NextDecisionSteps, NextTerminalSteps
            continue      #continue next while loop without increase step
        
        #else this and next decision steps includes all agents, collect (s, a, r, s1)
        for idx in range(len(DecisionSteps)):
            #find decision agents and record their state, value and ACTIONS
            AgentID = DecisionSteps.agent_id[idx]
            STATES[step][AgentID]=s[idx]
            VALUES[step][AgentID]=value[idx]
            ACTIONS[step][AgentID]=a[idx]
            LOG_PROBS[step][AgentID]=log_prob[idx]

        #collect reward of this action from next decision and terminal steps
        if(len(NextTerminalSteps) >0):
            #if next step has terminal agents, then collect terminal agents first
            Collect_REWARDS_and_MASKS(step, NextTerminalSteps, 0)
        else:  #else collect r and next state from decision steps
            Collect_REWARDS_and_MASKS(step, NextDecisionSteps, 1)
        
        if(print_message and (step % 500)==0):
            print(step, end=",")
        step = step + 1
        DecisionSteps, TerminalSteps = NextDecisionSteps, NextTerminalSteps
    print()


def compute_gae(next_value):
    value1 = VALUES + [next_value.cpu()]
    gae = 0
    returns = []
    for step in reversed(range(INTERACTION_STEPS )):
        delta = REWARDS[step] + GAMMA*value1[step + 1]*MASKS[step]-value1[step]
        gae = delta + GAMMA*LAMBD*MASKS[step]*gae
        returns.insert(0, gae + VALUES[step])
    return returns


def ppo_iter():
    buffer_size = MERGED_STATES.size(0)
    for _ in range(buffer_size// BATCH_SIZE ):
        rand_ids = np.random.randint(0, buffer_size, BATCH_SIZE )
        yield MERGED_STATES[rand_ids, :], MERGED_ACTIONS[rand_ids, :], MERGED_NEXT_STATES[rand_ids, :],              MERGED_LOG_PROBS[rand_ids, :], MERGED_RETURNS[rand_ids, :], MERGED_ADVANTAGES[rand_ids, :]


def ppo_update():
    #print("epoch:")
    for epoch in range(N_EPOCH):
        #print(epoch, end = ", ")
        for b_s, b_a, b_s_, b_old_LOG_PROBS, b_return, b_advantage in ppo_iter():
            dist, value = NET(b_s.to(device))       
            critic_loss = (b_return.to(device) - value).pow(2).mean()
            entropy = dist.entropy().mean()
            b_a_new = dist.sample()
            b_new_LOG_PROBS = dist.log_prob(b_a_new)
            ratio = (b_new_LOG_PROBS - b_old_LOG_PROBS.to(device)).exp()
            surr1 = ratio * b_advantage.to(device)
            surr2 = torch.clamp(ratio, 1.0-EPSILON, 1.0+EPSILON) * b_advantage.to(device)
            actor_loss  = - torch.min(surr1, surr2).mean()
            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()
    return float(critic_loss), float(actor_loss)


# # Main
NET = Net().to(device)
LEARNING_RATE = 0.0003
OPTIMIZER = optim.Adam(NET.parameters(), lr=LEARNING_RATE )

N_AGENTS = 10  #The number of training scenes in Unity 
INTERACTION_STEPS = 254  #(Walker.yaml 2048) buffer_size=N_AGENTS * INTERACTION_STEPS=20480

a = torch.FloatTensor([[0]]*N_AGENTS ) 
b = torch.FloatTensor([[0]*N_ACTIONS]*N_AGENTS ) 
c = torch.FloatTensor([[0]*N_STATES]*N_AGENTS ) 

VALUES =REWARDS = MASKS = [a]*INTERACTION_STEPS
LOG_PROBS = ACTIONS = [b]*INTERACTION_STEPS
STATES = NEXT_STATES = [c]*INTERACTION_STEPS

GAMMA = 0.995
LAMBD = 0.95
BATCH_SIZE = 254         #Walker.yaml 2048
BETA = 0.005
EPSILON = 0.2
N_EPOCH = 3
MAX_STEPS = 30000    #Walker.yaml 30M
SUMMARY_FREQ = 3000  #Walker.yaml 30K
TIME_HORIZON = 1000  #I do not use this parameter in my porgram

print("Please press play in Unity editor")
ENV = UnityEnvironment(file_name= None, base_port=5004)
ENV.reset()
BEHAVIOR_NAME = list(ENV.behavior_specs.keys())
BEHAVIOR_NAME = BEHAVIOR_NAME[0]

ActorLossLst = []
CriticLossLst = []
ForwardLossLst = []
steps  = 0 
summary = SUMMARY_FREQ

while (steps < MAX_STEPS):
    #print("Collecting ", INTERACTION_STEPS, " training steps from ", N_AGENTS, " agents", end=": ")
    collect_training_data(print_message=True)
    _, next_value = NET(NEXT_STATES[-1].to(device)) 
    
    print("Compute GAE of these training data set")
    RETURNS = compute_gae(next_value)
    MERGED_RETURNS   = torch.cat(RETURNS).detach()
    MERGED_LOG_PROBS = torch.cat(LOG_PROBS).detach()
    MERGED_VALUES    = torch.cat(VALUES).detach()
    MERGED_STATES    = torch.cat(STATES) 
    MERGED_NEXT_STATES    = torch.cat(NEXT_STATES) 
    MERGED_ACTIONS   = torch.cat(ACTIONS)
    MERGED_ADVANTAGES = MERGED_RETURNS - MERGED_VALUES
    
    print("Optimize NN with PPO")
    critic_loss, actor_loss = ppo_update()
    CriticLossLst.append(critic_loss)
    ActorLossLst.append(actor_loss)
    if(steps > summary):
        print("Already train ", steps, " steps") 
        print("Critic loss = ", round(critic_loss, 2), " Actor loss = ", round(actor_loss, 2))
        summary += SUMMARY_FREQ

    steps += INTERACTION_STEPS*N_AGENTS

ENV.close()

