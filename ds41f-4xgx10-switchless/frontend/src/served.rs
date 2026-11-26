//! The settings pico serves with, as the defaults of `dsfront serve`: a start with no settings of these runs the
//! measured configuration. Each can still be given another value; `--no-thinking` turns thinking off. What describes one
//! machine (the tokenizer's path, the node samplers' and media worker's addresses, the fan tool) is not here.

use crate::encoding::Encoding;

pub const MODEL_NAME: &str = "deepseek-v4.1-flash-1017";
pub const ENCODING: Encoding = Encoding::V41_1017;
pub const THINKING: bool = true;
pub const EFFORT: i64 = 75;
pub const MAX_TOKENS: u32 = 1_048_576;
pub const TEMPERATURE: f32 = 1.0;
pub const TOP_P: f32 = 0.95;
/// The most memory the process may hold, in MiB: the same number as the engine's `--front-mib`.
pub const FRONT_MIB: usize = 840;
/// Seconds a request waits while the engine is down (the ring lent, or a restart) before a 503 (`--hold-s`).
pub const HOLD_S: u64 = 900;

/// Thinking for requests that do not say: on, unless `--no-thinking` is given. `--thinking` spells the default out; when
/// both are given, the later one wins.
pub fn default_thinking(args: &[String]) -> bool {
    args.iter().rev().find(|a| *a == "--thinking" || *a == "--no-thinking").map_or(THINKING, |a| a == "--thinking")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    /// The served list, as the settings file holds it (FRONT_ARGS without the machine's own values).
    #[test]
    fn the_served_list() {
        assert_eq!(MODEL_NAME, "deepseek-v4.1-flash-1017");
        assert_eq!(ENCODING, Encoding::parse("1017").unwrap());
        assert_eq!(EFFORT, 75);
        assert_eq!(MAX_TOKENS, 1_048_576);
        assert_eq!(TEMPERATURE, 1.0);
        assert_eq!(TOP_P, 0.95);
        assert_eq!(FRONT_MIB, 840);
        assert_eq!(HOLD_S, 900);
        assert!(default_thinking(&args("dsfront serve")), "thinking on unless turned off");
    }

    #[test]
    fn thinking_flags() {
        assert!(default_thinking(&args("dsfront serve --thinking")));
        assert!(!default_thinking(&args("dsfront serve --no-thinking")));
        assert!(default_thinking(&args("dsfront serve --no-thinking --thinking")), "the later flag wins");
        assert!(!default_thinking(&args("dsfront serve --thinking --no-thinking --model-name m")));
    }
}
