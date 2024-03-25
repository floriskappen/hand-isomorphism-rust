
mod constants;
mod hand_indexer;
mod hand_indexer_state;
mod deck;


use crate::hand_indexer::HandIndexer;



fn main() {
    let hand_indexer_preflop = HandIndexer::new(1, &[2]).unwrap();

    if let Some(size) = hand_indexer_preflop.hand_indexer_size(0) {
        println!("Preflop hand indexer size: {}", size);
    }
}
