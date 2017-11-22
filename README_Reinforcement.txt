reward = self.mdp.getReward(state,action,next_State) (s,a,s')

Used this function to compute Q value

Vk[s] = ∑s' P(s'|s,a) (R(s,a,s')+ γVk-1[s'])
