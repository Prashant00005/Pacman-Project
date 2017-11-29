reward = self.mdp.getReward(state,action,next_State) (s,a,s')

Used this function to compute Q value

Vk[s] = ∑s' P(s'|s,a) (R(s,a,s')+ γVk-1[s'])


For question2 in analysis.py

#V(s) <- [max] γ ∑ P(s'|s,a) V(s') + R(s)
           a

Update for qlearningAgents.py

Q(s,a) <- (1-α) * Q(s,a) + α * [ reward + γ * max Q(s',a) ]
                                               a


approximate Q-function
Q(s,a)=∑i=1nfi(s,a)wi



difference=(r+γmaxa′Q(s′,a′))−Q(s,a)

Updating Weight vectors
wi←wi+α⋅difference⋅fi(s,a)
