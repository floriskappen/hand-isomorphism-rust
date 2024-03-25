use std::sync::Mutex;
use lazy_static::lazy_static;

// Constants
pub const MAX_GROUP_INDEX: usize = 0x100000;
pub const MAX_CARDS_PER_ROUND: usize = 15;
pub const MAX_ROUNDS: usize = 8;
pub const ROUND_SHIFT: u32 = 4;
pub const ROUND_MASK: u32 = 0xf;
pub const RANKS: usize = 13; // Assuming a standard deck
pub const SUITS: usize = 4; // Assuming a standard deck
pub const CARDS: usize = 52; // Assuming a standard deck

// Static data initialization using lazy_static for thread-safe, one-time initialization
lazy_static! {
    pub static ref NTH_UNSET: Vec<Vec<u8>> = {
        let mut nth_unset = vec![vec![0xff; RANKS]; 1 << RANKS];
        for i in 0..(1 << RANKS) {
            let mut set = !i & ((1 << RANKS) - 1);
            for j in 0..RANKS {
                nth_unset[i][j] = if set != 0 { set.trailing_zeros() as u8 } else { 0xff };
                if set != 0 {
                    set &= set - 1;
                }
            }
        }
        nth_unset
    };
    pub static ref EQUAL: Vec<Vec<bool>> = {
        let mut equal = vec![vec![false; SUITS]; 1 << (SUITS - 1)];
        for i in 0..(1 << (SUITS - 1)) {
            for j in 1..SUITS {
                equal[i][j] = (i & (1 << (j - 1))) != 0;
            }
        }
        equal
    };
    pub static ref NCR_RANKS: Mutex<Vec<Vec<u64>>> = Mutex::new({
        let mut ncr_ranks = vec![vec![0; RANKS + 1]; RANKS + 1];
        ncr_ranks[0][0] = 1;
        for i in 1..=RANKS {
            ncr_ranks[i][0] = 1;
            ncr_ranks[i][i] = 1;
            for j in 1..i {
                ncr_ranks[i][j] = ncr_ranks[i - 1][j - 1] + ncr_ranks[i - 1][j];
            }
        }
        ncr_ranks
    });

    pub static ref NCR_GROUPS: Mutex<Vec<Vec<u64>>> = Mutex::new({
        let mut ncr_groups = vec![vec![0u64; SUITS + 1]; MAX_GROUP_INDEX];
        ncr_groups[0][0] = 1;
        for i in 1..MAX_GROUP_INDEX {
            ncr_groups[i][0] = 1;
            // println!("{} {}", MAX_GROUP_INDEX, SUITS);
            // panic!();
            if i < SUITS + 1 {
                ncr_groups[i][i] = 1;
            }
            for j in 1..=i.min(SUITS) {
                // wrapping_add to prevent overflow warning in debug mode. But in reality we never get this.
                ncr_groups[i][j] = ncr_groups[i - 1][j - 1].wrapping_add(ncr_groups[i - 1][j]);
            }
        }
        ncr_groups
    });
    pub static ref RANK_SET_TO_INDEX: Vec<u64> = {
        let mut rank_set_to_index = vec![0; 1 << RANKS];
        for i in 0..(1 << RANKS) {
            let mut set = i as u32;
            let mut j = 1;
            while set != 0 {
                let rank = set.trailing_zeros();
                let ncr_ranks = NCR_RANKS.lock().unwrap();
                rank_set_to_index[i] += ncr_ranks[rank as usize][j];
                j += 1;
                set &= set - 1;
            }
        }
        rank_set_to_index
    };
    pub static ref INDEX_TO_RANK_SET: Vec<Vec<u32>> = {
        let mut index_to_rank_set = vec![vec![0; 1 << RANKS]; RANKS + 1];
        for i in 0..(1 << RANKS) {
            let index = i as u32;
            let popcount = index.count_ones() as usize;
            let index = RANK_SET_TO_INDEX[i];
            index_to_rank_set[popcount][index as usize] = i as u32;
        }
        index_to_rank_set
    };
    pub static ref SUIT_PERMUTATIONS: Mutex<Vec<Vec<u32>>> = Mutex::new({
        let num_permutations = factorial(SUITS);
        let mut permutations = vec![vec![0; SUITS]; num_permutations];
        for i in 0..num_permutations {
            let mut index = i;
            let mut used = 0;
            for j in 0..SUITS {
                let suit = index % (SUITS - j);
                index /= SUITS - j;
                let shifted_suit = NTH_UNSET[used as usize][suit as usize];
                permutations[i][j] = shifted_suit as u32;
                used |= 1 << shifted_suit;
            }
        }
        permutations
    });
}

// Utility functions, like factorial, used in initialization
fn factorial(n: usize) -> usize {
    (1..=n).product()
}
