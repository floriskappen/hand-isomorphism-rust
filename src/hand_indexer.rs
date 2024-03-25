
use crate::constants::{
    SUITS,
    MAX_ROUNDS,
    CARDS,
    ROUND_SHIFT,
    ROUND_MASK,
    RANKS,

    NCR_RANKS,
    NCR_GROUPS
};

#[derive(PartialEq, Clone, Copy)]
enum Action {
    CountConfiguations,
    TabulateConfigurations,
    CountPermutations,
    TabulatePermutations
}

pub struct HandIndexer {
    cards_per_round: [u8; MAX_ROUNDS],
    round_start: [usize; MAX_ROUNDS],
    rounds: usize,
    configurations: Vec<usize>,
    permutations: [usize; MAX_ROUNDS],
    round_size: [usize; MAX_ROUNDS],

    permutation_to_configuration: Vec<Vec<usize>>,
    permutation_to_pi: Vec<Vec<usize>>,
    configuration_to_equal: Vec<Vec<usize>>,
    configuration: Vec<Vec<[usize; SUITS]>>,
    configuration_to_suit_size: Vec<Vec<[usize; SUITS]>>,
    configuration_to_offset: Vec<Vec<usize>>,
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
            j += cards as usize;
        }

        // Count configurations
        indexer.enumerate_configurations(Action::CountConfiguations);
        
        // Allocate space based on counted configurations and then tabulate configurations
        for i in 0..rounds {
            indexer.configuration_to_equal[i] = vec![0; indexer.configurations[i]];
            indexer.configuration_to_offset[i] = vec![0; indexer.configurations[i]];
            indexer.configuration[i] = vec![[0; SUITS]; indexer.configurations[i]];
            indexer.configuration_to_suit_size[i] = vec![[0; SUITS]; indexer.configurations[i]];
        }

        // Tabulate configurations
        indexer.enumerate_configurations(Action::TabulateConfigurations);

        // Count permutations
        indexer.enumerate_configurations(Action::CountPermutations);

        // Allocate space based on counted permutations and then tabulate permutations
        for i in 0..rounds {
            indexer.permutation_to_configuration[i] = vec![0; indexer.permutations[i]];
            indexer.permutation_to_pi[i] = vec![0; indexer.permutations[i]];
        }

        // Tabulate permutations
        indexer.enumerate_configurations(Action::TabulatePermutations);

        Some(indexer)
    }

    fn count_configurations(&mut self, round: usize) {
        if round >= self.configurations.len() {
            self.configurations.resize(round + 1, 0);
        }

        self.configurations[round] += 1;
        // Additional logic as needed
    }

    fn tabulate_configurations(&mut self, round: usize, configuration: &[usize; SUITS]) {
        if round >= self.configurations.len() {
            self.configurations.resize_with(round + 1, Default::default);
            self.configuration.resize_with(round + 1, Default::default);
            self.configuration_to_suit_size.resize_with(round + 1, Default::default);
            self.configuration_to_offset.resize_with(round + 1, Default::default);
            self.configuration_to_equal.resize_with(round + 1, Default::default);
        }

        let ncr_ranks = NCR_RANKS.lock().unwrap();
        let ncr_groups = NCR_GROUPS.lock().unwrap();

        // Find the insertion point to keep configurations sorted
        let pos = self.configuration[round].iter().position(|&c| c > *configuration).unwrap_or(self.configuration[round].len());

        // Insert at the found position, shifting elements if necessary
        self.configuration[round].insert(pos, *configuration);
        self.configuration_to_suit_size[round].insert(pos, [0; SUITS]);
        self.configuration_to_offset[round].insert(pos, 0);
        self.configuration_to_equal[round].insert(pos, 0);

        // Update configurations count
        self.configurations[round] += 1;

        // Calculate 'configuration_to_suit_size', 'configuration_to_offset', and 'configuration_to_equal'
        for (i, &rank_set) in configuration.iter().enumerate() {
            let mut size: u64 = 1;
            let mut remaining = RANKS;

            for bit in rank_set.to_le_bytes() {
                let ranks = bit.count_ones() as usize;
                size *= ncr_ranks[remaining][ranks] as u64;
                remaining -= ranks;
            }

            self.configuration_to_suit_size[round][pos][i] = size as usize;

            let equal_mask = if i > 0 && configuration[i] == configuration[i - 1] { 1 << i } else { 0 };
            self.configuration_to_equal[round][pos] |= equal_mask;
        }

        let total_size: u64 = self.configuration_to_suit_size[round][pos].iter().map(|&x| x as u64).product();
        self.configuration_to_offset[round][pos] = ncr_groups[total_size as usize][SUITS] as usize;

        // Assuming calculation of 'configuration_to_offset' and 'configuration_to_equal' uses 'NCR_RANKS' and 'NCR_GROUPS'
    }

    fn count_permutations(&mut self, round: usize, count: &[usize]) {
        let mut idx = 0;
        let mut mult = 1;
        for i in 0..=round {
            let remaining = self.cards_per_round[i] as usize;
            for j in 0..SUITS - 1 {
                let shift_amount = (self.rounds - i - 1) * ROUND_SHIFT as usize;
                let size = (count[j] >> shift_amount) & ROUND_MASK as usize;
                idx += mult * size;
                mult *= remaining + 1 - j; // Adjust for zero-based indexing of `j` vs. one-based in original
            }
        }
        self.permutations[round] = self.permutations[round].max(idx + 1);
    }

    fn tabulate_permutations(&mut self, round: usize, count: &[usize; SUITS]) {
        let mut idx = 0;
        let mut mult = 1;
        // Calculate the index for the permutation based on `count`
        for i in 0..=round {
            let remaining = self.cards_per_round[i] as usize;
            for j in 0..SUITS-1 {
                let shift_amount = (self.rounds - i - 1) * ROUND_SHIFT as usize;
                let size = (count[j] >> shift_amount) & ROUND_MASK as usize;
                idx += mult * size;
                mult *= remaining + 1 - j;
            }
        }

        // Prepare permutation storage if necessary
        if self.permutation_to_pi.len() <= round {
            self.permutation_to_pi.resize_with(round + 1, Default::default);
            self.permutation_to_configuration.resize_with(round + 1, Default::default);
        }

        // Sort `count` by its values to generate pi (permutation index)
        let mut pi: Vec<usize> = (0..SUITS).collect();
        pi.sort_unstable_by_key(|&i| count[i]);
        let pi_idx = pi.iter().enumerate().fold((0, 1), |(acc, mult), (i, &pi_i)| {
            let smaller = pi[..i].iter().filter(|&&pi_j| pi_j < pi_i).count();
            (acc + (pi_i - smaller) * mult, mult * (SUITS - i))
        }).0;

        // Ensure storage for the new indices
        self.permutation_to_pi[round].resize(idx + 1, 0);
        self.permutation_to_pi[round][idx] = pi_idx;

        // Binary search to find the configuration that matches `count`, sorted by `pi`
        let pos = self.configuration[round].binary_search_by(|conf| {
            conf.iter().enumerate().map(|(i, &conf_i)| {
                let pi_pos = pi.iter().position(|&pi_i| pi_i == i).unwrap();
                conf_i.cmp(&count[pi_pos])
            }).find(|&ord| ord != std::cmp::Ordering::Equal).unwrap_or(std::cmp::Ordering::Equal)
        }).unwrap_or_else(|e| e); // Use the Err value as insertion point if not found

        // Ensure storage for the new configuration mapping
        self.permutation_to_configuration[round].resize(idx + 1, 0);
        self.permutation_to_configuration[round][idx] = pos;
    }

    fn enumerate_configurations(&mut self, action: Action) {
        let rounds = self.rounds;
        let mut used: [usize; SUITS] = [0; SUITS];
        let mut configuration: [usize; SUITS] = [0; SUITS];
        let cards_per_round_copy = self.cards_per_round;
        // Assuming `cards_per_round` has been properly initialized
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
                Action::CountPermutations => {
                    self.count_permutations(round, configuration);
                }
                Action::TabulatePermutations => {
                    self.tabulate_permutations(round, configuration);
                }
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

    pub fn hand_indexer_size(&self, round: usize) -> Option<usize> {
        if round < self.rounds {
            Some(self.round_size[round])
        } else {
            None
        }
    }
}

