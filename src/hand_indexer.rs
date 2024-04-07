use crate::constants::{
    CARDS, INDEX_TO_RANK_SET, MAX_ROUNDS, NCR_GROUPS, NCR_RANKS, EQUAL, NTH_UNSET, RANKS, RANK_SET_TO_INDEX, ROUND_MASK, ROUND_SHIFT, SUITS, SUIT_PERMUTATIONS
};
use crate::deck::{deck_get_rank, deck_get_suit, Card, deck_make_card};
use crate::hand_indexer_state::HandIndexerState;

#[derive(PartialEq, Clone, Copy)]
enum Action {
    CountConfiguations,
    TabulateConfigurations,
    CountPermutations,
    TabulatePermutations
}

pub struct HandIndexer {
    pub cards_per_round: [u8; MAX_ROUNDS],
    round_start: [u64; MAX_ROUNDS],
    rounds: usize,
    configurations: Vec<u64>,
    permutations: [u64; MAX_ROUNDS],
    round_size: [u64; MAX_ROUNDS],

    permutation_to_configuration: Vec<Vec<u64>>,
    permutation_to_pi: Vec<Vec<u64>>,
    configuration_to_equal: Vec<Vec<u64>>,
    configuration: Vec<Vec<[usize; SUITS]>>,
    configuration_to_suit_size: Vec<Vec<[u64; SUITS]>>,
    configuration_to_offset: Vec<Vec<u64>>,
}

impl HandIndexer {
    pub fn new(rounds: usize, cards_per_round: &[u8]) -> Option<Self> {
        if rounds == 0 || rounds > MAX_ROUNDS {
            return None;
        }
        let mut count = 0;
        for &cards in cards_per_round.iter().take(rounds) {
            count += cards as usize;
            if count > CARDS {
                return None;
            }
        }

        let mut indexer = HandIndexer {
            cards_per_round: [0; MAX_ROUNDS],
            round_start: [0; MAX_ROUNDS],
            rounds,
            permutations: [0; MAX_ROUNDS],
            round_size: [0; MAX_ROUNDS],
            
            configurations: vec![0; MAX_ROUNDS],
            permutation_to_configuration: vec![vec![]; rounds],
            permutation_to_pi: vec![vec![]; rounds],
            configuration_to_equal: vec![vec![]; rounds],
            configuration: vec![vec![]; rounds],
            configuration_to_suit_size: vec![vec![]; rounds],
            configuration_to_offset: vec![vec![]; rounds],
        };

        // Copy the cards_per_round information
        indexer.cards_per_round[..rounds].copy_from_slice(&cards_per_round[..rounds]);

        // Calculate the starting index of each round
        let mut j = 0;
        for (i, &cards) in cards_per_round.iter().enumerate().take(rounds) {
            indexer.round_start[i] = j;
            j += cards as u64;
        }

        // Count configurations
        indexer.enumerate_configurations(Action::CountConfiguations);
        
        // Allocate space based on counted configurations and then tabulate configurations
        for i in 0..rounds {
            indexer.configuration_to_equal[i] = vec![0; indexer.configurations[i] as usize];
            indexer.configuration_to_offset[i] = vec![0; indexer.configurations[i] as usize];
            indexer.configuration[i] = vec![[0; SUITS]; indexer.configurations[i] as usize];
            indexer.configuration_to_suit_size[i] = vec![[0; SUITS]; indexer.configurations[i] as usize];
        }
        
        for i in 0..indexer.configurations.len() {
            indexer.configurations[i] = 0;
        }
        // Tabulate configurations
        indexer.enumerate_configurations(Action::TabulateConfigurations);

        for i in 0..rounds {
            let mut accum = 0u64;
            for j in 0..indexer.configurations[i] as usize {
                let next = accum + indexer.configuration_to_offset[i][j];
                indexer.configuration_to_offset[i][j] = accum;
                accum = next;
            }

            indexer.round_size[i] = accum;
        }

        // Count permutations
        indexer.enumerate_permutations(Action::CountPermutations);

        // Allocate space based on counted permutations and then tabulate permutations
        for i in 0..rounds {
            indexer.permutation_to_configuration[i] = vec![0; indexer.permutations[i] as usize];
            indexer.permutation_to_pi[i] = vec![0; indexer.permutations[i] as usize];
        }

        // Tabulate permutations
        indexer.enumerate_permutations(Action::TabulatePermutations);

        Some(indexer)
    }

    fn count_configurations(&mut self, round: usize) {
        if round >= self.configurations.len() {
            self.configurations.resize(round + 1, 0);
        }

        self.configurations[round] += 1;
    }

    fn tabulate_configurations(&mut self, round: usize, configuration: &[usize; SUITS]) {
        // Increment configurations count for this round and get the current configuration ID
        let mut id = self.configurations[round] as usize;
        self.configurations[round] += 1;

        // Extend vectors to ensure they are large enough to hold the new configuration
        self.configuration[round].resize(id + 1, [0; SUITS]);
        self.configuration_to_suit_size[round].resize(id + 1, [0; SUITS]);
        self.configuration_to_offset[round].resize(id + 1, 0);
        self.configuration_to_equal[round].resize(id + 1, 0);


        // Insertion sort logic adapted from C
        while id > 0 {
            let prev_id = id - 1;
            if configuration < &self.configuration[round][prev_id] {
                // Shift configurations one position to make room for the new configuration
                self.configuration[round][id] = self.configuration[round][prev_id];
                self.configuration_to_suit_size[round][id] = self.configuration_to_suit_size[round][prev_id];
                self.configuration_to_offset[round][id] = self.configuration_to_offset[round][prev_id];
                self.configuration_to_equal[round][id] = self.configuration_to_equal[round][prev_id];
                id -= 1;
            } else {
                break;
            }
        }


        // Insert the new configuration at the correct position
        self.configuration[round][id] = *configuration;
        self.configuration_to_offset[round][id] = 1;

        // Calculation loop for suit sizes and offsets
        let mut equal: u64 = 0;
        let mut i = 0usize;
        while i < SUITS {
            let mut size = 1u64;
            let mut remaining = RANKS;

            // Correctly calculate ranks using ROUND_SHIFT and ROUND_MASK for each suit
            for j in 0..=round {
                let ranks = (configuration[i] >> (ROUND_SHIFT * (self.rounds - j - 1) as u32)) & ROUND_MASK as usize;
                size *= NCR_RANKS[remaining][ranks];
                remaining -= ranks;
            }

            // Find the next suit index j that does not equal the current configuration[i]
            let j = (i + 1..SUITS).find(|&j| configuration[j] != configuration[i]).unwrap_or(SUITS);
            // Update suit sizes for all suits from i to j
            for k in i..j {
                self.configuration_to_suit_size[round][id][k] = size;
            }
  
            // Multiply the configuration offset by the number of groups determined by nCr_groups
            self.configuration_to_offset[round][id] *= NCR_GROUPS[(size as usize) + j - i - 1][j - i];

            // Set equal bits for suits from i + 1 to j
            for k in i + 1..j {
                equal |= 1 << k;
            }

            i = j;
        }

        self.configuration_to_equal[round][id] = equal >> 1;
    }

    fn count_permutations(&mut self, round: usize, count: &[u32]) {
        let mut idx: u32 = 0;
        let mut mult: u32 = 1;
        for i in 0..=round {
            let mut remaining: u32 = self.cards_per_round[i] as u32;
            for j in 0..SUITS - 1 {
                let size = ((count[j] >> (((self.rounds - i - 1) as u32) * ROUND_SHIFT)) as u32) & ROUND_MASK;
                idx += mult * size;
                mult *= remaining + 1;
                remaining -= size;
            }
        }
        
        if self.permutations[round] < (idx + 1) as u64 {
            self.permutations[round] = (idx + 1) as u64;
        }
    }

    fn tabulate_permutations(&mut self, round: usize, count: &[u32]) {
        let mut idx = 0;
        let mut mult = 1;

        for i in 0..=round {
            let mut remaining: u32 = self.cards_per_round[i] as u32;
            for j in 0..SUITS - 1 {
                let size = ((count[j] >> (((self.rounds - i - 1) as u32) * ROUND_SHIFT)) as u32) & ROUND_MASK;
                idx += mult * size;
                mult *= remaining + 1;
                remaining -= size;
            }
        }

        let mut pi: Vec<usize> = (0..SUITS).collect();

        for i in 1..SUITS {
            let mut j = i;
            let pi_i = pi[i];
            while j > 0 && count[pi_i] > count[pi[j - 1]] {
                pi[j] = pi[j - 1];
                j -= 1;
            }
            pi[j] = pi_i;
        }

        
        let mut pi_idx: u32 = 0;
        let mut pi_mult: u32 = 1;
        let mut pi_used: u32 = 0;
        
        for i in 0..SUITS {
            let this_bit = 1 << pi[i];
            let smaller = (this_bit - 1) & pi_used;
            let smaller = smaller.count_ones();
            pi_idx += (pi[i] as u32 - smaller) * pi_mult;
            pi_mult *= SUITS as u32 - i as u32;
            pi_used |= this_bit;
        }

        self.permutation_to_pi[round][idx as usize] = pi_idx as u64;

        let mut low: u64 = 0;
        let mut high: u64 = self.configurations[round]; // Assuming configurations[round] is a Vec<Vec<u32>>

        while low < high {
            let mid = (low + high) / 2;

            let mut compare = 0;
            for i in 0..SUITS {
                let this = count[pi[i] as usize] as usize;
                let other = self.configuration[round][mid as usize][i]; // Assuming configuration is a Vec<Vec<Vec<u32>>>

                if other > this {
                    compare = -1;
                    break;
                } else if other < this {
                    compare = 1;
                    break;
                }
            }

            if compare == -1 {
                high = mid;
            } else if compare == 0 {
                low = mid;
                high = mid;
                break; // Since low and high are set to the same value, we can exit the loop.
            } else {
                low = mid + 1;
            }
        }

        self.permutation_to_configuration[round][idx as usize] = low;
    }

    fn enumerate_configurations(&mut self, action: Action) {
        let rounds = self.rounds;
        let mut used: [usize; SUITS] = [0; SUITS];
        let mut configuration: [usize; SUITS] = [0; SUITS];
        let cards_per_round_copy = self.cards_per_round;
        self.enumerate_configurations_r(
            rounds,
            cards_per_round_copy,
            0,
            self.cards_per_round[0] as usize,
            0,
            (1 << SUITS) - 2,
            &mut used,
            &mut configuration,
            action,
        );
    }

    fn enumerate_configurations_r(
        &mut self,
        rounds: usize,
        cards_per_round: [u8; 8],
        round: usize,
        remaining: usize,
        suit: usize,
        equal: usize,
        used: &mut [usize; SUITS],
        configuration: &mut [usize; SUITS],
        action: Action
    ) {
        if suit == SUITS {
            match action {
                Action::CountConfiguations => {
                    self.count_configurations(round);
                }
                Action::TabulateConfigurations => {
                    self.tabulate_configurations(round, configuration);
                }
                default => {}
            }

            if round + 1 < rounds {
                self.enumerate_configurations_r(rounds, cards_per_round, round + 1, cards_per_round[round + 1] as usize, 0, equal, used, configuration, action);
            }
        } else {
            let min = if suit == SUITS - 1 { remaining } else { 0 };
            let mut max = RANKS - used[suit];
            if remaining < max {
                max = remaining;
            }

            let mut previous = RANKS + 1;
            let was_equal = equal & (1 << suit) != 0;
            if was_equal {
                previous = (configuration[suit - 1] >> (ROUND_SHIFT * (rounds - round - 1) as u32)) & ROUND_MASK as usize;
                if previous < max {
                    max = previous;
                }
            }

            let old_configuration = configuration[suit];
            let old_used = used[suit];
            for i in min..=max {
                let new_configuration = old_configuration | i << (ROUND_SHIFT * (rounds - round - 1) as u32);
                let new_equal = (equal & !(1 << suit)) | ((was_equal && i == previous) as usize) << suit;

                used[suit] = old_used + i;
                configuration[suit] = new_configuration;
                self.enumerate_configurations_r(rounds, cards_per_round, round, remaining - i, suit + 1, new_equal, used, configuration, action);
                configuration[suit] = old_configuration;
                used[suit] = old_used;
            }
        }
    }

    fn enumerate_permutations_r(
        &mut self,
        rounds: usize,
        cards_per_round: [u8; 8],
        round: usize,
        remaining: usize,
        suit: usize,
        used: &mut [usize; SUITS],
        count: &mut [u32],
        action: Action,
    ) {
        if suit == SUITS {
            match action {
                Action::CountPermutations => {
                    self.count_permutations(round, count);
                }
                Action::TabulatePermutations => {
                    self.tabulate_permutations(round, count);
                }
                default => {}
            }

            if round + 1 < rounds {
                self.enumerate_permutations_r(rounds, cards_per_round, round + 1, cards_per_round[round + 1] as usize, 0, used, count, action);
            }
        } else {
            let min = if suit == SUITS - 1 { remaining } else { 0 };
            let mut max = RANKS - used[suit];
            if remaining < max {
                max = remaining;
            }

            let old_count = count[suit];
            let old_used = used[suit];
            for i in min..=max {
                let new_count: u32 = old_count | (i << (ROUND_SHIFT * (rounds - round - 1) as u32)) as u32;

                used[suit] = old_used + i;
                count[suit] = new_count;
                self.enumerate_permutations_r(rounds, cards_per_round, round, remaining - i, suit + 1, used, count, action);
                count[suit] = old_count;
                used[suit] = old_used;
            }
        }
    }

    fn enumerate_permutations(&mut self, action: Action) {
        let mut used: [usize; SUITS] = [0; SUITS];
        let mut count: [u32; SUITS] = [0; SUITS];
        let cards_per_round_copy = self.cards_per_round;
        // Assuming `cards_per_round` has been properly initialized
        self.enumerate_permutations_r(
            self.rounds,
            cards_per_round_copy,
            0,
            self.cards_per_round[0] as usize,
            0,
            &mut used,
            &mut count,
            action,
        );
    }
    pub fn hand_index_next_round(&self, cards: &[Card], state: &mut HandIndexerState) -> usize {
        let round = state.round;
        state.round += 1;

        if round >= self.rounds {
            panic!("hand_index_next_round: state.round >= self.rounds");
        }

        let mut ranks = [0; SUITS];
        let mut shifted_ranks = [0; SUITS];
        
        for i in 0..self.cards_per_round[round] as usize {
            assert!(cards[i] < CARDS as Card, "Invalid card.");

            let rank = deck_get_rank(cards[i]) as usize;
            let suit = deck_get_suit(cards[i]) as usize;
            let rank_bit = 1 << rank;

            assert!(ranks[suit as usize] & rank_bit == 0, "Rank bit already set.");

            ranks[suit] |= rank_bit;
            shifted_ranks[suit as usize] |= rank_bit >> ((rank_bit - 1) & state.used_ranks[suit as usize]).count_ones();
        }
        
        for i in 0..SUITS {
            // Ensure there are no duplicate cards
            assert!(state.used_ranks[i] & ranks[i] == 0, "Duplicate cards detected.");

            let used_size = state.used_ranks[i].count_ones() as usize;
            let this_size = ranks[i].count_ones() as usize;

            state.suit_index[i] += state.suit_multiplier[i] * RANK_SET_TO_INDEX[shifted_ranks[i] as usize] as usize;
            state.suit_multiplier[i] *= NCR_RANKS[RANKS - used_size][this_size] as usize;
            state.used_ranks[i] |= ranks[i];
        }

        let mut remaining = self.cards_per_round[round] as usize;

        for i in 0..SUITS-1 {
            let this_size = ranks[i].count_ones() as usize;
            state.permutation_index += state.permutation_multiplier * this_size;
            state.permutation_multiplier *= remaining + 1;
            remaining -= this_size;
        }
        
        let configuration = self.permutation_to_configuration[round][state.permutation_index];
        let pi_index = self.permutation_to_pi[round][state.permutation_index];
        let equal_index = self.configuration_to_equal[round][configuration as usize];
        let offset = self.configuration_to_offset[round][configuration as usize];

        let suit_permutations = SUIT_PERMUTATIONS.lock().unwrap();
        let pi = &suit_permutations[pi_index as usize];

        let mut suit_index = vec![0; SUITS];
        let mut suit_multiplier = [0; SUITS];
        
        for i in 0..SUITS {
            suit_index[i] = state.suit_index[pi[i] as usize];
            suit_multiplier[i] = state.suit_multiplier[pi[i] as usize];
        }

        // SORTING
        let mut index = offset; // Assuming 'offset' is already defined as hand_index_t
        let mut multiplier = 1; // hand_index_t
        
        let mut i = 0;
        while i < SUITS {
            let mut part = 0;
            let mut size = 0;
        
            if i + 1 < SUITS && EQUAL[equal_index as usize][i + 1] {
                if i + 2 < SUITS && EQUAL[equal_index as usize][i + 2] {
                    if i + 3 < SUITS && EQUAL[equal_index as usize][i + 3] {
                        // Four equal suits
                        suit_index.swap(i, i + 1);
                        suit_index.swap(i + 2, i + 3);
                        suit_index.swap(i, i + 2);
                        suit_index.swap(i + 1, i + 3);
                        suit_index.swap(i + 1, i + 2);
                        part = (suit_index[i] as u64) + NCR_GROUPS[suit_index[i + 1] + 1][2] + NCR_GROUPS[suit_index[i + 2] + 2][3] + NCR_GROUPS[suit_index[i + 3] + 3][4];
                        size = NCR_GROUPS[suit_multiplier[i] + 3][4];
                        i += 4;
                    } else {
                        // Three equal suits
                        suit_index.swap(i, i + 1);
                        suit_index.swap(i, i + 2);
                        suit_index.swap(i + 1, i + 2);
                        part = (suit_index[i] as u64) + NCR_GROUPS[suit_index[i + 1] + 1][2] + NCR_GROUPS[suit_index[i + 2] + 2][3];
                        size = NCR_GROUPS[suit_multiplier[i] + 2][3];
                        i += 3;
                    }
                } else {
                    // Two equal suits
                    suit_index.swap(i, i + 1);
                    part = (suit_index[i] as u64) + NCR_GROUPS[suit_index[i + 1] + 1][2];
                    size = NCR_GROUPS[suit_multiplier[i] + 1][2];
                    i += 2;
                }
            } else {
                // No equal suits
                part = suit_index[i] as u64;
                size = suit_multiplier[i] as u64;
                i += 1;
            }
        
            index += multiplier * part;
            multiplier *= size;
        }
        index as usize // Return the final hand index
    }

    pub fn hand_unindex(&self, round: usize, mut index: u64, cards: &mut Vec<Card>) -> bool {
        if round >= self.rounds || index >= self.round_size[round] {
            return false;
        }

        let mut low: usize = 0;
        let mut high: usize = self.configurations[round] as usize;
        let mut configuration_idx = 0;


        while low < high {
            let mid = (low + high) / 2;
            if self.configuration_to_offset[round][mid] <= index {
                configuration_idx = mid;
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        index -= self.configuration_to_offset[round][configuration_idx];

        let mut suit_index = [0; SUITS];

        let mut i = 0;
        while i < SUITS {
            let mut j = i + 1;
            while j < SUITS && self.configuration[round][configuration_idx][j] == self.configuration[round][configuration_idx][i] {
                j += 1;
            }

            let suit_size = self.configuration_to_suit_size[round][configuration_idx][i] as usize;
            let group_size = NCR_GROUPS[suit_size + j - i - 1][j - i];
            let mut group_index = index % group_size;
            index /= group_size;
    
            while i < j - 1 {
                low = (
                    f64::exp(
                        f64::ln(group_index as f64)/((j-i) as f64) - 1f64 + f64::ln((j-i) as f64)
                    ) - (j as f64) - (i as f64)
                ).floor() as usize;

                suit_index[i] = low;

                high = (
                    f64::exp(
                        f64::ln(group_index as f64)/((j-i) as f64) + f64::ln((j-i) as f64)
                    ) - (j as f64) + (i as f64) + 1f64
                ).ceil() as usize;

                if high > suit_size {
                    high = suit_size;
                }
                if high <= low {
                    low = 0;
                }
    
                while low < high {
                    let mid = (low + high) / 2;
                    if NCR_GROUPS[mid + j - i - 1][j - i] <= group_index {
                        suit_index[i] = mid;
                        low = mid + 1;
                    } else {
                        high = mid;
                    }
                }
    
                group_index -= NCR_GROUPS[suit_index[i] + j - i - 1][j - i];

                i+=1;
            }
    
            suit_index[i] = group_index as usize;

            i+=1;            
        }


        // Initialize location with round_start values
        let mut location = [0u64; MAX_ROUNDS];
        location[..self.rounds].copy_from_slice(&self.round_start[..self.rounds]);

        for i in 0..SUITS {
            let mut used = 0u32;
            let mut m = 0usize;
            
            for j in 0..self.rounds as usize {
                
                let n = (self.configuration[round][configuration_idx][i] >> (ROUND_SHIFT * (self.rounds - j - 1) as u32)) & ROUND_MASK as usize;

                let round_size = NCR_RANKS[RANKS - m][n] as usize;
                m += n;
                
                let round_index = suit_index[i] % round_size;
                suit_index[i] /= round_size;
                
                let mut shifted_cards = INDEX_TO_RANK_SET[n][round_index]; // Adapt INDEX_TO_RANK_SET access
                let mut rank_set = 0u32;
        
                for k in 0..n {
                    let shifted_card = 1 << shifted_cards.trailing_zeros();
                    shifted_cards ^= shifted_card;
                    let card_rank = shifted_card.trailing_zeros() as u32; // Get the rank from shifted_card
                    let card = NTH_UNSET[used as usize][card_rank as usize];

                    rank_set |= 1 << card; // Update rank_set with this card
                    cards[location[j] as usize + k] = deck_make_card(i, card as usize); // Assign the card to cards array

                }
                location[j] += n as u64;
                used |= rank_set;


            }
        }


        true
    }

    pub fn hand_indexer_size(&self, round: usize) -> Option<u64> {
        if round < self.rounds {
            Some(self.round_size[round])
        } else {
            None
        }
    }

    // UNUSED & UNTESTED
    pub fn hand_index_all(&self, cards: &[Card]) -> Vec<usize> {
        let mut indices = vec![0; self.rounds];
        let mut state = HandIndexerState::new(); // Replace with the actual state initialization

        for i in 0..self.rounds {
            let start = self.round_start[i] as usize;
            let end = start + self.cards_per_round[i] as usize;
            // Make sure to handle potential out-of-bounds here for `cards`
            if end <= cards.len() {
                indices[i] = self.hand_index_next_round(&cards[start..end], &mut state);
            }
        }

        indices
    }

    // UNUSED & UNTESTED
    pub fn hand_index_last(&self, cards: &[Card]) -> usize {
        let indices = self.hand_index_all(cards);
        *indices.last().unwrap_or(&0) // Safely return the last element or 0 if not present
    }

    pub fn hand_to_canonical_representation(&self, cards: Vec<Card>) -> Vec<Card> {
        let round: usize;
        let mut state = HandIndexerState::new();
        if cards.len() == 2 {
            round = 0;
        } else if cards.len() == 5 {
            round = 1;
        } else if cards.len() == 6 {
            round = 2;
        } else {
            round = 3;
        }

        let mut total_cards = 0;
        for i in 0..=round {
            self.hand_index_next_round(&cards[total_cards..], &mut state);
            total_cards += self.cards_per_round[i] as usize;
        }

        let configuration = self.permutation_to_configuration[round][state.permutation_index as usize];
        let pi_index = self.permutation_to_pi[round][state.permutation_index as usize] as usize;
        let offset = self.configuration_to_offset[round][configuration as usize];

        let suit_permutations = SUIT_PERMUTATIONS.lock().unwrap();
        let pi = &suit_permutations[pi_index as usize];

        let mut suit_index = [0; SUITS];
        let mut suit_multiplier = [0; SUITS];
        for i in 0..SUITS {
            suit_index[i] = state.suit_index[pi[i] as usize];
            suit_multiplier[i] = state.suit_multiplier[pi[i] as usize];
        }

        let mut index = offset;
        let mut multiplier = 1;
        let mut i = 0;

        while i < SUITS {
            let mut j = i + 1;
            while j < SUITS && self.configuration[round][configuration as usize][j] == self.configuration[round][configuration as usize][i] {
                j += 1;
            }
            let suit_size = self.configuration_to_suit_size[round][configuration as usize][i] as usize;
            let group_size = NCR_GROUPS[suit_size + j - i - 1][j - i];
            let mut group_index = 0;
            while i < j - 1 {
                group_index += NCR_GROUPS[suit_index[i] as usize + j - i - 1][j - i];
                i += 1;
            }
            group_index += suit_index[i] as u64;
            index += multiplier * group_index;
            multiplier *= group_size;
            i += 1;
        }

        let mut canonicalized_cards = vec![0u8; 7];
        self.hand_unindex(round, index, &mut canonicalized_cards);

        return canonicalized_cards[..cards.len()].to_vec();
    }
}
