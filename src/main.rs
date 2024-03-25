
mod constants;
mod hand_indexer;
mod hand_indexer_state;
mod deck;


use crate::hand_indexer::HandIndexer;



fn main() {
    let hand_indexer_preflop = HandIndexer::new(4, &[2, 3, 1, 1]).unwrap();

    if let Some(size) = hand_indexer_preflop.hand_indexer_size(3) {
        println!("Preflop hand indexer size: {}", size);
    }
}
