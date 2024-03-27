
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

mod constants;
mod hand_indexer;
mod hand_indexer_state;
mod deck;

use crate::hand_indexer::HandIndexer;
use crate::deck::{RANK_TO_CHAR, SUIT_TO_CHAR, deck_get_suit, deck_get_rank};

fn generate_canonical_hands(indexer: &HandIndexer, round: usize) -> io::Result<()> {
    let size = indexer.hand_indexer_size(round).unwrap() as usize;
    let mut cards = vec![0u32; 7]; // Adjust the size according to your needs
    let mut total_cards = 0u8;

    for i in 0..=round {
        total_cards += indexer.cards_per_round[i as usize];
    }

    // Allocate memory for the canonical hands using a Vec
    let mut canonical_hands: Vec<Vec<u32>> = Vec::with_capacity(size as usize);

    // Generate canonical hands
    for i in 0..size {
        indexer.hand_unindex(round, i as u64, &mut cards, false);
        canonical_hands.push(cards[..total_cards as usize].to_vec());
    }

    // Write to file
    let mut file = File::create(Path::new("hands.txt"))?;
    for hand in canonical_hands.iter() {
        let hand_string = hand.iter()
            .map(|&card| {
                // println!("{} {} {}", card, RANK_TO_CHAR[deck_get_rank(card) as usize], SUIT_TO_CHAR[deck_get_suit(card) as usize]);
                return format!("{}{} ", RANK_TO_CHAR[deck_get_rank(card) as usize], SUIT_TO_CHAR[deck_get_suit(card) as usize])
            })
            .collect::<String>();
        writeln!(file, "{}", hand_string)?;
    }

    // Print the first 10 canonical hands
    // println!("First 10 canonical hands for round {}:", round);
    // for (i, hand) in canonical_hands.iter().take(10).enumerate() {
    //     // print!("Hand {}: ", i);
    //     for &card in hand {
    //         print!("{} ", card);
    //         // print!("{}{} ", RANK_TO_CHAR[deck_get_rank(card) as usize], SUIT_TO_CHAR[deck_get_suit(card) as usize]);
    //     }
    //     println!();
    // }

    println!("Canonical hands!: {}", canonical_hands.len());

    Ok(())
}

fn main() {
    // River indexer
    let hand_indexer = HandIndexer::new(4, &[2, 3, 1, 1]).unwrap();

    generate_canonical_hands(&hand_indexer, 0);
}
