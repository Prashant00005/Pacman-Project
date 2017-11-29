Used this function to compute Q value (refer file valueIterationAgents.py line 103)

Vk[s] = ∑s' P(s'|s,a) (R(s,a,s')+ γVk-1[s'])


Q-Value update  (refer file qlearningAgents.py line 169)

Q(s,a) <- (1-α) * Q(s,a) + α * [ reward + γ * max Q(s',a) ]
                                               a

Approximate Q-function (refer file qlearningAgents.py line 244)

        n
Q(s,a)= ∑  fi (s,a) wi
        i=1


Updating Weight vectors (refer file qlearningAgents.py line 274)

wi ← wi + α * difference * fi(s,a)

difference = (r + γ max′ Q(s′,a′) ) − Q(s,a)
                     a
