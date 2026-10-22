//! Animated pet companion widget for the input bar.
//!
//! Ported from the `flash` agent TUI. Renders a small 3-row ASCII pet beside
//! the textarea that reacts to agent events (Thinking/Coding/Searching/…),
//! blinks periodically, and falls asleep when idle.
//!
//! The [`Pet`] enum and [`random_pet`] live here (not in CLI config) so the
//! widget is self-contained. CLI config holds only an `Option<String>` that
//! parses into a [`Pet`] via [`Pet::from_name`].

use std::time::{SystemTime, UNIX_EPOCH};

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
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs().rotate_left(32) ^ u64::from(duration.subsec_nanos()))
        .unwrap_or(0x9E37_79B9_7F4A_7C15)
        .wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// Pick a random pet using a minimal xorshift RNG seeded from the clock.
pub fn random_pet() -> Pet {
    Pet::ALL[random_below(&mut time_seed().max(1), Pet::ALL.len())]
}

// All callers use bounds below 256, so one random byte is sufficient.
fn random_below(state: &mut u64, upper: usize) -> usize {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    let byte = state.wrapping_mul(0x2545_F491_4F6C_DD1D).to_be_bytes()[0];
    usize::from(byte) % upper
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

pub struct PetWidget {
    pub kind: Pet,
    state: PetState,
    pending_state: Option<PetState>,
    frame: usize,
    tick: usize,
    state_tick: usize,
    frame_ticks: usize,
    random_state: u64,
    last_state_change: usize,
    idle_ticks: usize,
    blinking: bool,
    blink_start: usize,
    next_blink: usize,
}

impl PetWidget {
    pub fn new(kind: Pet) -> Self {
        Self::with_seed(kind, time_seed())
    }

    fn with_seed(kind: Pet, seed: u64) -> Self {
        let mut pet = Self {
            kind,
            state: PetState::Idle,
            pending_state: None,
            frame: 0,
            tick: 0,
            state_tick: 0,
            frame_ticks: 0,
            random_state: seed.max(1),
            last_state_change: 0,
            idle_ticks: 0,
            blinking: false,
            blink_start: 0,
            next_blink: 0,
        };
        pet.schedule_frame();
        pet.schedule_blink();
        pet
    }

    /// Activity changes have a short cooldown independent of pose duration.
    fn min_state_ticks(&self) -> usize {
        match self.state {
            PetState::Idle | PetState::Running => 10,
            PetState::Searching => 15,
            PetState::Coding => 15,
            PetState::Reading => 15,
            PetState::Sleeping => 30,
            _ => STATE_TICK_MIN,
        }
    }

    fn tempo(&self) -> usize {
        match self.kind {
            Pet::Cat => 2,
            Pet::Dog | Pet::Octopus | Pet::Robot => 3,
            Pet::Duck | Pet::Penguin | Pet::Dragon => 4,
            Pet::Ghost | Pet::Axolotl => 5,
            Pet::Capybara => 7,
        }
    }

    fn schedule_frame(&mut self) {
        let tempo = self.tempo();
        self.frame_ticks = match self.state {
            PetState::Idle => match self.frame {
                0 => 20 + tempo * 5 + random_below(&mut self.random_state, tempo * 8),
                2 => tempo,
                5 | 8 => tempo * 3,
                _ => tempo * 2,
            },
            PetState::Running => tempo * 2,
            PetState::Searching if self.frame == 0 => tempo * 2,
            PetState::Searching | PetState::Coding => tempo * 3,
            PetState::Thinking => tempo * 2,
            PetState::Reading => tempo * 8,
            PetState::Sleeping => tempo * 10,
            PetState::Blink => BLINK_DURATION,
        };
    }

    fn advance_frame(&mut self) {
        self.frame = match self.state {
            PetState::Idle => match self.frame {
                // Choose a short alternate gesture about once in four cycles.
                0 if random_below(&mut self.random_state, 4) == 0 => 6,
                0 => 1,
                5 | 8 => 0,
                _ => self.frame + 1,
            },
            PetState::Running | PetState::Searching => {
                // The preparation pose plays once, followed by the action loop.
                let next = self.frame + 1;
                if next >= pet_frames(self.kind, self.state).len() {
                    1
                } else {
                    next
                }
            }
            _ => (self.frame + 1) % pet_frames(self.kind, self.state).len(),
        };
        self.state_tick = 0;
        self.schedule_frame();
    }

    fn schedule_blink(&mut self) {
        self.next_blink =
            self.tick + BLINK_MIN + random_below(&mut self.random_state, BLINK_MAX - BLINK_MIN);
    }

    fn can_blink(&self) -> bool {
        matches!(self.state, PetState::Idle) && self.frame == 0 || self.state == PetState::Reading
    }

    pub fn set_state(&mut self, state: PetState) {
        // A newer request supersedes any activity waiting for its first frame.
        self.pending_state = None;
        if self.state == state {
            return;
        }

        if state != PetState::Idle
            && self.state != PetState::Sleeping
            && self.tick - self.last_state_change < self.min_state_ticks()
        {
            self.pending_state = Some(state);
            return;
        }

        self.change_state(state);
    }

    fn change_state(&mut self, state: PetState) {
        self.state = state;
        self.frame = 0;
        self.state_tick = 0;
        self.last_state_change = self.tick;
        self.idle_ticks = 0;
        self.blinking = false;
        self.schedule_frame();
        self.schedule_blink();
    }

    /// Reset the sleep timer without interrupting the agent's activity.
    pub(super) fn wake_on_input(&mut self) {
        self.idle_ticks = 0;
        if self.state == PetState::Sleeping {
            self.change_state(PetState::Idle);
        }
    }

    pub fn tick(&mut self) {
        self.tick += 1;
        if !self.blinking {
            self.state_tick += 1;
        }

        if self.pending_state.is_some()
            && self.tick - self.last_state_change >= self.min_state_ticks()
            && let Some(state) = self.pending_state.take()
        {
            self.change_state(state);
        }

        if !self.blinking && self.state_tick >= self.frame_ticks {
            self.advance_frame();
        }

        if self.blinking {
            if self.tick - self.blink_start >= BLINK_DURATION {
                self.blinking = false;
                self.schedule_blink();
            }
        } else if self.tick >= self.next_blink && self.can_blink() {
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
        if self.blinking {
            let blink_frames = pet_frames(self.kind, PetState::Blink);
            return blink_frames[0];
        }

        let frames = pet_frames(self.kind, self.state);
        frames[self.frame % frames.len()]
    }

    pub fn render(&self, buf: &mut Buffer, area: Rect) {
        let frame = self.current_frame();
        let style = self.style();
        let x_offset = area.x + area.width.saturating_sub(pet_art::FRAME_WIDTH + 2);

        for (row_idx, line) in frame.iter().enumerate() {
            let y = area.y + row_idx as u16;

            if y >= area.y + area.height {
                break;
            }

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
    fn idle_holds_rest_then_moves_quickly() {
        let mut pet = PetWidget::with_seed(Pet::Cat, 1);
        pet.next_blink = usize::MAX;
        let rest = pet_frames(Pet::Cat, PetState::Idle)[0];
        let duration = pet.frame_ticks;
        for _ in 1..duration {
            pet.tick();
            assert_eq!(pet.current_frame(), rest);
        }
        pet.tick();
        assert_ne!(pet.frame, 0);
        assert!(pet.frame_ticks < duration);
    }

    #[test]
    fn capybara_moves_more_slowly_than_cat() {
        let mut cat = PetWidget::with_seed(Pet::Cat, 1);
        let mut capybara = PetWidget::with_seed(Pet::Capybara, 1);
        assert!(capybara.frame_ticks > cat.frame_ticks);
        cat.change_state(PetState::Running);
        capybara.change_state(PetState::Running);
        assert!(capybara.frame_ticks > cat.frame_ticks);
    }

    #[test]
    fn seeded_idle_varies_pauses_and_gestures() {
        let mut primary = false;
        let mut alternate = false;
        let mut durations = Vec::new();
        for seed in 1..=32 {
            let mut pet = PetWidget::with_seed(Pet::Cat, seed);
            pet.next_blink = usize::MAX;
            durations.push(pet.frame_ticks);
            for _ in 0..pet.frame_ticks {
                pet.tick();
            }
            primary |= pet.frame == 1;
            alternate |= pet.frame == 6;
            assert!(matches!(pet.frame, 1 | 6));
            for _ in 0..100 {
                pet.tick();
                if pet.frame == 0 {
                    break;
                }
            }
            assert_eq!(pet.frame, 0, "Both gestures must settle back to rest.");
        }
        assert!(primary && alternate);
        assert!(durations.iter().any(|duration| *duration != durations[0]));
    }

    #[test]
    fn blink_waits_until_idle_gesture_finishes() {
        let mut pet = PetWidget::with_seed(Pet::Cat, 1);
        pet.frame = 3;
        pet.schedule_frame();
        pet.next_blink = 0;
        while pet.frame != 0 {
            pet.tick();
            if pet.frame != 0 {
                assert!(!pet.blinking);
                assert_eq!(
                    pet.current_frame(),
                    pet_frames(Pet::Cat, PetState::Idle)[pet.frame]
                );
            }
        }
        assert!(pet.blinking);
    }

    #[test]
    fn blink_pauses_pose_clock() {
        let mut pet = PetWidget::with_seed(Pet::Cat, 1);
        pet.state_tick = pet.frame_ticks - 1;
        pet.blinking = true;
        pet.blink_start = pet.tick;
        for _ in 0..BLINK_DURATION {
            pet.tick();
            assert_eq!(pet.frame, 0);
        }
        assert!(!pet.blinking);
        pet.tick();
        assert_ne!(pet.frame, 0);
    }

    #[test]
    fn preparation_plays_once_before_action_loop() {
        for kind in Pet::ALL {
            for state in [PetState::Running, PetState::Searching] {
                let mut pet = PetWidget::with_seed(*kind, 1);
                pet.change_state(state);
                assert_eq!(pet.current_frame(), pet_frames(*kind, state)[0]);
                for _ in 0..8 {
                    for _ in 0..pet.frame_ticks {
                        pet.tick();
                    }
                    assert_ne!(pet.frame, 0);
                    assert_eq!(pet.current_frame(), pet_frames(*kind, state)[pet.frame]);
                }
            }
        }
    }

    #[test]
    fn completion_interrupts_gesture_and_blink() {
        let mut pet = PetWidget::with_seed(Pet::Cat, 1);
        pet.change_state(PetState::Running);
        pet.set_state(PetState::Coding);
        pet.set_state(PetState::Idle);
        assert_eq!(pet.current_frame(), pet_frames(Pet::Cat, PetState::Idle)[0]);
        pet.blinking = true;
        pet.set_state(PetState::Thinking);
        for _ in 0..pet.min_state_ticks() {
            pet.tick();
        }
        assert!(!pet.blinking);
        assert_eq!(
            pet.current_frame(),
            pet_frames(Pet::Cat, PetState::Thinking)[0]
        );
    }

    #[test]
    fn latest_activity_survives_cooldown() {
        let mut pet = PetWidget::new(Pet::Cat);
        pet.set_state(PetState::Searching);
        pet.set_state(PetState::Coding);
        for _ in 0..pet.min_state_ticks() {
            pet.tick();
        }
        assert_eq!(pet.state, PetState::Coding);
        assert_eq!(pet.frame, 0);
        assert_eq!(pet.state_tick, 0);
    }

    #[test]
    fn idle_immediately_clears_pending_activity() {
        let mut pet = PetWidget::new(Pet::Cat);
        for _ in 0..pet.min_state_ticks() {
            pet.tick();
        }
        pet.set_state(PetState::Running);
        pet.set_state(PetState::Coding);
        pet.set_state(PetState::Idle);
        assert_eq!(pet.state, PetState::Idle);
        for _ in 0..30 {
            pet.tick();
        }
        assert_eq!(pet.state, PetState::Idle);
    }

    #[test]
    fn latest_request_can_keep_current_activity() {
        let mut pet = PetWidget::new(Pet::Cat);
        for _ in 0..pet.min_state_ticks() {
            pet.tick();
        }
        pet.set_state(PetState::Running);
        pet.set_state(PetState::Coding);
        pet.set_state(PetState::Running);
        for _ in 0..30 {
            pet.tick();
        }
        assert_eq!(pet.state, PetState::Running);
    }

    #[test]
    fn activity_wakes_pet_immediately_after_sleep_starts() {
        let mut pet = PetWidget::new(Pet::Cat);
        for _ in 0..SLEEP_THRESHOLD {
            pet.tick();
        }
        pet.set_state(PetState::Thinking);
        assert_eq!(pet.state, PetState::Thinking);
        assert_eq!(pet.frame, 0);
    }

    #[test]
    fn typing_wakes_pet_and_restarts_sleep_timer() {
        let mut pet = PetWidget::new(Pet::Cat);
        for _ in 0..SLEEP_THRESHOLD {
            pet.tick();
        }
        pet.wake_on_input();
        assert_eq!(pet.state, PetState::Idle);
        for _ in 0..SLEEP_THRESHOLD - 1 {
            pet.tick();
        }
        pet.wake_on_input();
        pet.tick();
        assert_eq!(pet.state, PetState::Idle);
    }

    #[test]
    fn typing_preserves_active_and_pending_work() {
        let mut pet = PetWidget::new(Pet::Cat);
        for _ in 0..pet.min_state_ticks() {
            pet.tick();
        }
        pet.set_state(PetState::Running);
        pet.set_state(PetState::Coding);
        pet.wake_on_input();
        assert_eq!(pet.state, PetState::Running);
        for _ in 0..pet.min_state_ticks() {
            pet.tick();
        }
        assert_eq!(pet.state, PetState::Coding);
    }

    #[test]
    fn idle_gestures_stay_inside_pet_column() {
        for kind in Pet::ALL {
            let mut pet = PetWidget::new(*kind);
            let area = Rect::new(2, 1, 14, 3);
            let mut buf = Buffer::empty(Rect::new(0, 0, 18, 5));
            let frames = pet_frames(*kind, PetState::Idle);
            assert!(frames.len() > 2, "{kind:?} needs an idle gesture");
            for index in 0..frames.len() {
                pet.frame = index;
                buf.reset();
                pet.render(&mut buf, area);
                for y in 0..5 {
                    for x in 0..18 {
                        if !area.contains((x, y).into()) {
                            assert_eq!(buf[(x, y)].symbol(), " ");
                        }
                    }
                }
            }
        }
    }

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
                match state {
                    PetState::Idle => assert_eq!(frames.len(), 9),
                    PetState::Running | PetState::Searching => assert_eq!(frames.len(), 3),
                    _ => {}
                }

                for frame in frames {
                    assert_eq!(frame.len(), 3, "{:?} {:?} frame not 3 rows", pet, state);
                    for row in *frame {
                        assert!(row.is_ascii());
                        assert!(row.len() <= usize::from(pet_art::FRAME_WIDTH));
                    }
                }
            }
        }
    }
}
