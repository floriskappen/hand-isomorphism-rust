
// Define card as a type alias for u32 for better readability.
pub type Card = u32;

// Mapping arrays could be implemented in Rust using slices or arrays,
// but for simplicity, we'll just define functions that mimic the C constants behavior.
// Assuming you would like to convert ranks and suits to characters.

/// Returns the character representation of a card's rank.
pub fn rank_to_char(rank: Card) -> char {
    "23456789TJQKA".chars().nth(rank as usize).unwrap_or('?')
}

/// Returns the character representation of a card's suit.
pub fn suit_to_char(suit: Card) -> char {
    "♠♥♦♣".chars().nth(suit as usize).unwrap_or('?')
}

/// Extracts the suit from a card value.
pub fn deck_get_suit(card: Card) -> Card {
    card & 3
}

/// Extracts the rank from a card value.
pub fn deck_get_rank(card: Card) -> Card {
    card >> 2
}

/// Constructs a card value from suit and rank.
pub fn deck_make_card(suit: Card, rank: Card) -> Card {
    (rank << 2) | suit
}
