mod constants;
mod hand_indexer;
mod hand_indexer_state;
mod deck;
mod database;

use constants::BATCH_SIZE;
use crate::hand_indexer::HandIndexer;
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
    let mut total: u64 = 0;

    let session = create_session().await;

    // Generate canonical hands
    for i in 0..size {
        indexer.hand_unindex(round, i as u64, &mut cards, false);
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

        
        if i > 0 && i % BATCH_SIZE == 0 {
            println!("Inserting batch {}...", (i / BATCH_SIZE)+1);
            insert_batch(&session, canonical_hands.clone(), &database_table).await;
            canonical_hands.clear();
        }
        total += 1;
    }

    if canonical_hands.len() > 0 {
        println!("Inserting final batch...");
        insert_batch(&session, canonical_hands.clone(), &database_table).await;
        canonical_hands.clear();
    }
}

#[tokio::main]
async fn main() {
    // River indexer
    let hand_indexer = HandIndexer::new(4, &[2, 3, 1, 1]).unwrap();

    generate_canonical_hands_and_insert_into_database(&hand_indexer, 2).await;

    // let card = deck_make_card(1, 11); // Example card, King of Hearts
    // println!("card u32: {} rank: {} suit: {} encoded: {}", card, RANK_TO_CHAR[deck_get_rank(card) as usize], SUIT_TO_CHAR[deck_get_suit(card) as usize], card as u8);
}
