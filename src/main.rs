
mod constants;
mod hand_indexer;
mod hand_indexer_state;

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



fn main() {
    // println!("NTH_UNSET[0][0]: {:?}", NTH_UNSET[0][0]);
    // println!("EQUAL[1][1]: {:?}", EQUAL[1][1]);
    // println!("NCR_RANKS[5][3]: {:?}", NCR_RANKS[5][3]);
    // println!("NCR_GROUPS[5][3]: {:?}", NCR_GROUPS[5][3]);
    // println!("RANK_SET_TO_INDEX[16]: {:?}", RANK_SET_TO_INDEX[16]);
    // println!("INDEX_TO_RANK_SET[2][10]: {:?}", INDEX_TO_RANK_SET[2][10]);

    // // Accessing a Mutex-guarded global static variable safely
    // let suit_permutations = SUIT_PERMUTATIONS.lock().unwrap(); // Handle potential poisoning
    // println!("SUIT_PERMUTATIONS[0]: {:?}", suit_permutations[0]);

    // Example usage
    let rounds = 2;
    let cards_per_round = vec![3, 2]; // Example data
    let mut data = 0; // Placeholder for user-defined data
    // enumerate_configurations(rounds, &cards_per_round, |round, config, _data| {
    //     println!("Round: {}, Configuration: {:?}", round, config);
    // }, &mut data);
}
