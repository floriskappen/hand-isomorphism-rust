mod constants;
mod hand_indexer;
mod hand_indexer_state;
mod deck;

use deck::{card_from_string, card_to_string};
use crate::{deck::Card, hand_indexer::HandIndexer};

fn main() {
    // River indexer
    let hand_indexer = HandIndexer::new(4, &[2, 3, 1, 1]).unwrap();

    let cards: Vec<Card> = vec![
        card_from_string("Kh".to_string()),
        card_from_string("As".to_string()),
        card_from_string("Js".to_string()),
        card_from_string("Qs".to_string()),
        card_from_string("Td".to_string()),
        card_from_string("2d".to_string()),
        card_from_string("6c".to_string()),
    ];

    let canonical_hand = hand_indexer.hand_to_canonical_representation(&cards);
    let index = hand_indexer.hand_to_index(&cards);

    let canonical_hand_str = canonical_hand.iter()
        .map(|&card| card_to_string(card))
        .collect::<Vec<String>>()
        .join(",");

    println!("CANON: {}", canonical_hand_str);
    println!("INDEX: {}", index);
}
