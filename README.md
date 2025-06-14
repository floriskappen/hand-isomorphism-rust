# Poker Hand Isomorphism (Rust implementation)

Original author: Kevin Waugh
Original code: https://github.com/kdub0/hand-isomorphism

Kevin Waugh's paper which this code is based on: K. Waugh, 2013. A Fast and Optimal Hand Isomorphism Algorithm. In the Second Computer Poker and Imperfect Information Symposium at AAAI

---

I rewrote Kevin's poker hand isomorphism code fully in Rust. 

Kevin Waugh's poker hand isomorphism code is used to reduce the number of unique poker hand situations an AI needs to consider by grouping strategically equivalent hands together. This is called hand isomorphism, and it's essential for:
1. Reducing state space: Many poker hands are effectively the same when suits or player positions are permuted. Waugh’s code maps such hands to a canonical form, allowing the AI to treat them identically.
2. Efficient training and evaluation: By abstracting hands into fewer equivalence classes, algorithms (like Counterfactual Regret Minimization or DeepStack-style solvers) can operate on smaller, more general representations—making training faster and strategy generalization easier.
