# Open research questions

Paraphrased from the supplied specification; these are decisions to test or resolve with the advisor, not settled claims.

- Can DDPG's target actor be removed without harming stability or final return?
- Can independently optimized critics replace the ordinary online/target critic relationship while preserving a coherent Bellman recursion?
- Is current-input agreement, next-input agreement, or both necessary?
- Under which local assumptions could shared TD error act as a Lyapunov candidate?
- Can an acceptance/backtracking rule enforce minibatch descent while still permitting policy improvement?
- Is any parameter, memory, or runtime advantage measurable after fair comparison?
- After continuous-control validation, could the principle transfer to DQN?
- Should actor ascent use current replay states or next states from the board formulation?
- Should Critic Two fully replace a target critic, or does it need a soft-update component in an ablation?
- Should beta be linear in `Y`, quadratic in value outputs, or a critic weight regularizer?
- Should updates be simultaneous, alternating, or guarded, and should the guard cover the joint step or only the actor step?
- Is a prospective theorem about one minibatch, expected replay error, or convergence to a stationary point?
