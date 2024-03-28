
// Define card as a type alias for u32 for better readability.
pub type Card = u32;

// Mapping arrays could be implemented in Rust using slices or arrays,
// but for simplicity, we'll just define functions that mimic the C constants behavior.
// Assuming you would like to convert ranks and suits to characters.

// Returns the character representation of a card's rank.
pub const RANK_TO_CHAR: &[char] = &['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];

// Returns the character representation of a card's suit.
pub const SUIT_TO_CHAR: &[char] = &['s', 'h', 'd', 'c'];

// Extracts the suit from a card value.
pub fn deck_get_suit(card: Card) -> Card {
    card & 3
}

// Extracts the rank from a card value.
pub fn deck_get_rank(card: Card) -> Card {
    card >> 2
}

// Constructs a card value from suit and rank.
pub fn deck_make_card(suit: Card, rank: Card) -> Card {
    (rank << 2) | suit
}
