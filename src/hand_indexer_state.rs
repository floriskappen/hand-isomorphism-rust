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

pub struct HandIndexerState {
    pub suit_index: [usize; SUITS],
    pub suit_multiplier: [usize; SUITS],
    pub round: usize,
    pub permutation_index: usize,
    pub permutation_multiplier: usize,
    pub used_ranks: [u32; SUITS],
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
