mod constants;
mod hand_indexer;
mod hand_indexer_state;
mod deck;
mod database;

use deck::{card_from_string, card_to_string};
use crate::{deck::{deck_get_rank, deck_get_suit, Card, RANK_TO_CHAR, SUIT_TO_CHAR}, hand_indexer::HandIndexer};
use database::{create_session, insert_batch, DatabasePokerHand};

async fn generate_canonical_hands_and_insert_into_database(indexer: &HandIndexer, round: usize) {
    let size = indexer.hand_indexer_size(round).unwrap() as usize;
    let mut cards = vec![0u32; 7]; // Adjust the size according to your needs
    let mut total_cards = 0u8;

    for i in 0..=round {
        total_cards += indexer.cards_per_round[i as usize];
    }

    let database_table: String;
    match round {
        0 => database_table = "preflop_round".to_string(),
        1 => database_table = "flop_round".to_string(),
        2 => database_table = "turn_round".to_string(),
        default => database_table = "river_round".to_string(),
    }

    // Allocate memory for the canonical hands using a Vec
    let mut canonical_hands: Vec<DatabasePokerHand> = Vec::with_capacity(size as usize);
    let mut canonical_hands_ints: Vec<Vec<Card>> = vec![];
    let mut total: u64 = 0;

    let session = create_session().await;

    // Generate canonical hands
    for i in 0..size {
        indexer.hand_unindex(round, i as u64, &mut cards);
        let hand = cards[..total_cards as usize].to_vec();
        canonical_hands.push(
            DatabasePokerHand {
                cards_id: hand.iter()
                    .enumerate()
                    .map(|(i, &card)| {
                        if i == hand.len()-1 {
                            return format!("{}", card)
                        }
                        return format!("{},", card)
                    })
                    .collect::<String>()
            }
        );

        canonical_hands_ints.push(hand);
        
        // if i > 0 && i % BATCH_SIZE == 0 {
        //     println!("Inserting batch {}...", (i / BATCH_SIZE)+1);
        //     insert_batch(&session, canonical_hands.clone(), &database_table).await;
        //     canonical_hands.clear();
        // }
        total += 1;
    }

    for hand in canonical_hands_ints {
        println!(
            "{}{}, {}{}",
            RANK_TO_CHAR[deck_get_rank(hand[0]) as usize], SUIT_TO_CHAR[deck_get_suit(hand[0]) as usize],
            RANK_TO_CHAR[deck_get_rank(hand[1]) as usize], SUIT_TO_CHAR[deck_get_suit(hand[1]) as usize],
        )
    }
}

#[tokio::main]
async fn main() {
    // River indexer
    let hand_indexer = HandIndexer::new(4, &[2, 3, 1, 1]).unwrap();

    // println!("{}", card_from_string("Qh".to_string()));
    // println!("{}", card_to_string(41));
    let cards: Vec<Card> = vec![
        card_from_string("Kd".to_string()),
        card_from_string("Ah".to_string()),
        card_from_string("Qh".to_string()),
        card_from_string("Jh".to_string()),
        card_from_string("Tc".to_string()),
        card_from_string("2c".to_string()),
        card_from_string("6h".to_string()),
    ];

    let canonical_hand = hand_indexer.hand_to_canonical_representation(cards);
    let canonical_hand_str = canonical_hand.iter()
        .map(|&card| card_to_string(card))
        .collect::<Vec<String>>()
        .join(",");

    println!("CANON: {}", canonical_hand_str);
}
