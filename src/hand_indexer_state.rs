use crate::constants::{
    MAX_GROUP_INDEX,
    MAX_CARDS_PER_ROUND,
    ROUND_SHIFT,
    ROUND_MASK,
    RANKS,
    SUITS,
    NTH_UNSET,
    EQUAL,
    NCR_RANKS,
    NCR_GROUPS,
    RANK_SET_TO_INDEX,
    INDEX_TO_RANK_SET,
    SUIT_PERMUTATIONS,
};

struct HandIndexerState {
    suit_index: [usize; SUITS],
    suit_multiplier: [usize; SUITS],
    round: usize,
    permutation_index: usize,
    permutation_multiplier: usize,
    used_ranks: [u32; SUITS],
}

impl HandIndexerState {
    pub fn new() -> Self {
        HandIndexerState {
            suit_index: [0; SUITS],  // Initialized with 0s
            suit_multiplier: [1; SUITS],  // Each suit_multiplier initialized to 1
            round: 0,
            permutation_index: 0,
            permutation_multiplier: 1,  // Set to 1 as specified
            used_ranks: [0; SUITS],  // Initialized with 0s
        }
    }
}
