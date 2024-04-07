
// Define card as a type alias for u32 for better readability.
pub type Card = u8;

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
pub fn deck_make_card(suit: usize, rank: usize) -> Card {
    ((rank << 2) | suit) as Card
}

// Constructs a card from a string
pub fn card_from_string(card_string: String) -> Card {
    // Extract the rank and suit characters from the card string
    let mut chars = card_string.chars();
    let rank_char: char = chars.next().unwrap();
    let suit_char: char = chars.next().unwrap();

    // Find the index of the rank and suit characters in the respective arrays
    let rank_index = RANK_TO_CHAR.iter()
        .enumerate()
        .find(|(_, &ch)| ch == rank_char)
        .map(|(i, _)| i)
        .unwrap();
    let suit_index = SUIT_TO_CHAR.iter()
        .enumerate()
        .find(|(_, &ch)| ch == suit_char)
        .map(|(i, _)| i)
        .unwrap();

    return deck_make_card(suit_index, rank_index)
}

// Constructs a string from a card
pub fn card_to_string(card: Card) -> String {
        // Extract rank and suit from the card using provided functions
        let rank = deck_get_rank(card);
        let suit = deck_get_suit(card);

        return format!("{}{}", RANK_TO_CHAR[rank as usize], SUIT_TO_CHAR[suit as usize])
    
}
