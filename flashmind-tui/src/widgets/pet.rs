//! Animated pet companion widget for the input bar.
//!
//! Ported from the `flash` agent TUI. Renders a small 3-row ASCII pet beside
//! the textarea that reacts to agent events (Thinking/Coding/Searching/…),
//! blinks periodically, and falls asleep when idle.
//!
//! The [`Pet`] enum and [`random_pet`] live here (not in CLI config) so the
//! widget is self-contained. CLI config holds only an `Option<String>` that
//! parses into a [`Pet`] via [`Pet::from_name`].

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};

use super::pet_art;

const BLINK_DURATION: usize = 3;
const SLEEP_THRESHOLD: usize = 375;
const BLINK_MIN: usize = 63;
const BLINK_MAX: usize = 125;
const STATE_TICK_MIN: usize = 5;

/// Available pet kinds. Variant order must match the index in `pet_art::PETS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Pet {
    Cat = 0,
    Dog = 1,
    Duck = 2,
    Penguin = 3,
    Octopus = 4,
    Ghost = 5,
    Robot = 6,
    Axolotl = 7,
    Dragon = 8,
    Capybara = 9,
}

impl Pet {
    /// All pet kinds, in declaration order.
    pub const ALL: &'static [Pet] = &[
        Pet::Cat,
        Pet::Dog,
        Pet::Duck,
        Pet::Penguin,
        Pet::Octopus,
        Pet::Ghost,
        Pet::Robot,
        Pet::Axolotl,
        Pet::Dragon,
        Pet::Capybara,
    ];

    /// Parse a config string into a pet. Accepts lowercase names.
    /// Returns `None` for unknown names (caller falls back to random).
    pub fn from_name(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cat" => Some(Pet::Cat),
            "dog" => Some(Pet::Dog),
            "duck" => Some(Pet::Duck),
            "penguin" => Some(Pet::Penguin),
            "octopus" => Some(Pet::Octopus),
            "ghost" => Some(Pet::Ghost),
            "robot" => Some(Pet::Robot),
            "axolotl" => Some(Pet::Axolotl),
            "dragon" => Some(Pet::Dragon),
            "capybara" => Some(Pet::Capybara),
            _ => None,
        }
    }
}

/// A simple, dependency-free PRNG seeded from the wall clock.
fn time_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15)
        .wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// Pick a random pet using a minimal xorshift RNG seeded from the clock.
pub fn random_pet() -> Pet {
    let mut state = time_seed().max(1);
    let n = Pet::ALL.len();
    // xorshift64*
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    let idx = ((state.wrapping_mul(0x2545_F491_4F6C_DD1D)) >> 33) as usize % n;
    Pet::ALL[idx]
}

/// Get animation frames for this pet in the given state.
fn pet_frames(pet: Pet, state: PetState) -> &'static [&'static [&'static str]] {
    pet_art::PETS[pet as usize][state as usize]
}

/// Pet activity state. Variant order must match the state index in `pet_art::PETS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum PetState {
    Idle = 0,
    Thinking = 1,
    Searching = 2,
    Coding = 3,
    Running = 4,
    Reading = 5,
    Sleeping = 6,
    Blink = 7,
}

fn random_blink_interval(current_tick: usize) -> usize {
    let mut state = time_seed().max(1);
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    let span = (BLINK_MAX - BLINK_MIN) as u64;
    let r = ((state.wrapping_mul(0x2545_F491_4F6C_DD1D)) >> 33) % span.max(1);
    current_tick + BLINK_MIN + r as usize
}

pub struct PetWidget {
    pub kind: Pet,
    state: PetState,
    frame: usize,
    tick: usize,
    state_tick: usize,
    last_state_change: usize,
    idle_ticks: usize,
    blinking: bool,
    blink_start: usize,
    next_blink: usize,
}

impl PetWidget {
    pub fn new(kind: Pet) -> Self {
        Self {
            kind,
            state: PetState::Idle,
            frame: 0,
            tick: 0,
            state_tick: 0,
            last_state_change: 0,
            idle_ticks: 0,
            blinking: false,
            blink_start: 0,
            next_blink: random_blink_interval(0),
        }
    }

    /// Minimum ticks before the animation can advance to a new frame.
    fn min_state_ticks(&self) -> usize {
        match self.state {
            PetState::Running => 10,
            PetState::Searching => 15,
            PetState::Coding => 15,
            PetState::Reading => 15,
            PetState::Sleeping => 30,
            _ => STATE_TICK_MIN,
        }
    }

    pub fn set_state(&mut self, state: PetState) {
        if self.state == state {
            return;
        }

        if self.tick - self.last_state_change < self.min_state_ticks() {
            return;
        }

        self.state = state;
        self.frame = 0;
        self.last_state_change = self.tick;

        if !matches!(state, PetState::Idle | PetState::Sleeping) {
            self.idle_ticks = 0;
        }
    }

    pub fn tick(&mut self) {
        self.tick += 1;
        self.state_tick += 1;

        if self.state_tick >= self.min_state_ticks() {
            self.state_tick = 0;
            let frames = pet_frames(self.kind, self.state);
            self.frame = (self.frame + 1) % frames.len();
        }

        if self.blinking {
            if self.tick - self.blink_start >= BLINK_DURATION {
                self.blinking = false;
                self.next_blink = random_blink_interval(self.tick);
            }
        } else if self.tick >= self.next_blink
            && !matches!(self.state, PetState::Sleeping | PetState::Blink)
        {
            self.blinking = true;
            self.blink_start = self.tick;
        }

        if matches!(self.state, PetState::Idle) {
            self.idle_ticks += 1;

            if self.idle_ticks >= SLEEP_THRESHOLD {
                self.set_state(PetState::Sleeping);
            }
        }
    }

    pub fn current_frame(&self) -> &[&str] {
        if self.blinking
            && matches!(
                self.state,
                PetState::Idle | PetState::Thinking | PetState::Reading
            )
        {
            let blink_frames = pet_frames(self.kind, PetState::Blink);
            return blink_frames[0];
        }

        let frames = pet_frames(self.kind, self.state);
        frames[self.frame % frames.len()]
    }

    pub fn render(&self, buf: &mut Buffer, area: Rect) {
        let frame = self.current_frame();
        let style = self.style();

        for (row_idx, line) in frame.iter().enumerate() {
            let y = area.y + row_idx as u16;

            if y >= area.y + area.height {
                break;
            }

            let char_count = line.chars().count() as u16;
            let x_offset = if char_count < area.width {
                area.x + area.width - char_count - 2
            } else {
                area.x
            };

            for (col_idx, ch) in line.chars().enumerate() {
                if ch == ' ' {
                    continue;
                }

                let x = x_offset + col_idx as u16;

                if x >= area.x + area.width {
                    break;
                }

                if let Some(cell) = buf.cell_mut((x, y)) {
                    cell.set_char(ch);
                    cell.set_style(style);
                }
            }
        }
    }

    fn style(&self) -> Style {
        let color = match self.state {
            PetState::Idle => Color::Reset,
            PetState::Sleeping | PetState::Blink => Color::Gray,
            PetState::Thinking | PetState::Reading => Color::Cyan,
            PetState::Searching | PetState::Running => Color::Yellow,
            PetState::Coding => Color::Green,
        };

        Style::default().fg(color)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pet_from_name() {
        assert_eq!(Pet::from_name("cat"), Some(Pet::Cat));
        assert_eq!(Pet::from_name("Dragon"), Some(Pet::Dragon));
        assert_eq!(Pet::from_name("unknown"), None);
    }

    #[test]
    fn test_pet_auto_sleep() {
        let mut pet = PetWidget::new(Pet::Cat);

        for _ in 0..SLEEP_THRESHOLD {
            pet.tick();
        }

        assert_eq!(pet.state, PetState::Sleeping);
    }

    #[test]
    fn test_pet_wake_from_sleep() {
        let mut pet = PetWidget::new(Pet::Cat);

        for _ in 0..SLEEP_THRESHOLD {
            pet.tick();
        }

        assert_eq!(pet.state, PetState::Sleeping);

        for _ in 0..30 {
            pet.tick();
        }

        pet.set_state(PetState::Running);
        assert_eq!(pet.state, PetState::Running);
    }

    #[test]
    fn test_frame_cycling() {
        let mut pet = PetWidget::new(Pet::Cat);

        for _ in 0..20 {
            pet.tick();
        }
        pet.set_state(PetState::Running);

        let f0 = pet.frame;

        for _ in 0..10 {
            pet.tick();
        }

        assert_ne!(
            pet.frame, f0,
            "frame should have changed after ticks, state={:?}, f0={}",
            pet.state, f0
        );
    }

    #[test]
    fn test_current_frame_returns_3_rows() {
        let pet = PetWidget::new(Pet::Cat);
        assert_eq!(pet.current_frame().len(), 3);
    }

    #[test]
    fn test_all_pets_all_states() {
        let states = [
            PetState::Idle,
            PetState::Thinking,
            PetState::Searching,
            PetState::Coding,
            PetState::Running,
            PetState::Reading,
            PetState::Sleeping,
            PetState::Blink,
        ];

        for pet in Pet::ALL {
            for state in &states {
                let frames = pet_frames(*pet, *state);
                assert!(!frames.is_empty(), "{:?} {:?} has no frames", pet, state);

                for frame in frames {
                    assert_eq!(frame.len(), 3, "{:?} {:?} frame not 3 rows", pet, state);
                }
            }
        }
    }
}
