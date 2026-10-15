use std::fmt;

use flashmind_types::SessionUsage;
use flashmind_types::humanize::format_number;

pub(crate) struct UsageSummary<'a>(pub &'a SessionUsage);

impl fmt::Display for UsageSummary<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.tokens_unknown {
            writeln!(f, "Tokens used: input unavailable, output unavailable")?;
        } else if let Some(cached) = self.0.cached_input_tokens {
            writeln!(
                f,
                "Tokens used: {} input (+ {} cached), {} output",
                format_number(self.0.input_tokens.saturating_sub(cached)),
                format_number(cached),
                format_number(self.0.output_tokens),
            )?;
        } else {
            writeln!(
                f,
                "Tokens used: {} input (includes cached; breakdown unavailable), {} output",
                format_number(self.0.input_tokens),
                format_number(self.0.output_tokens),
            )?;
        }
        if self.0.cost_unknown {
            write!(f, "Cost: unavailable (pricing missing)")
        } else {
            write!(f, "Cost: ${}", self.0.total_cost.normalize())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flashmind_types::{ModelPricing, TokenUsage};
    use rust_decimal::Decimal;

    #[test]
    fn totals_survive_resume_and_keep_cost_precision() {
        let pricing = ModelPricing {
            prompt: Some(Decimal::new(2, 6)),
            completion: Some(Decimal::new(10, 6)),
            cache_read: Some(Decimal::new(2, 7)),
            ..Default::default()
        };
        let request = TokenUsage {
            prompt_tokens: 1000,
            completion_tokens: 20,
            cache_read_tokens: 800,
            ..Default::default()
        };
        let mut usage = SessionUsage::default();
        usage.add(&request, &pricing);
        let saved = serde_json::to_string(&usage).unwrap();
        let mut resumed: SessionUsage = serde_json::from_str(&saved).unwrap();
        resumed.add(&request, &pricing);
        assert_eq!(
            UsageSummary(&resumed).to_string(),
            "Tokens used: 400 input (+ 1,600 cached), 40 output\nCost: $0.00152"
        );
    }

    #[test]
    fn missing_pricing_is_not_reported_as_free() {
        let request = TokenUsage {
            prompt_tokens: 12,
            completion_tokens: 3,
            ..Default::default()
        };
        let mut usage = SessionUsage::default();
        usage.add(&request, &ModelPricing::default());
        assert_eq!(
            UsageSummary(&usage).to_string(),
            "Tokens used: 12 input (+ 0 cached), 3 output\nCost: unavailable (pricing missing)"
        );
        usage.add(
            &request,
            &ModelPricing {
                prompt: Some(Decimal::ZERO),
                completion: Some(Decimal::ZERO),
                ..Default::default()
            },
        );
        assert!(usage.cost_unknown);
    }

    #[test]
    fn older_totals_keep_the_cached_breakdown_unknown_after_resume() {
        let saved = r#"{
            "input_tokens": 2000,
            "output_tokens": 40,
            "total_cost": "0.00152",
            "tokens_unknown": false,
            "cost_unknown": false
        }"#;
        let mut usage: SessionUsage = serde_json::from_str(saved).unwrap();
        usage.add(
            &TokenUsage {
                prompt_tokens: 1000,
                completion_tokens: 20,
                cache_read_tokens: 800,
                ..Default::default()
            },
            &ModelPricing::default(),
        );
        assert_eq!(usage.cached_input_tokens, None);
        assert_eq!(
            UsageSummary(&usage).to_string(),
            "Tokens used: 3,000 input (includes cached; breakdown unavailable), 60 output\nCost: unavailable (pricing missing)"
        );
    }

    #[test]
    fn session_totals_can_exceed_request_counter_size() {
        let mut usage = SessionUsage::default();
        let request = TokenUsage {
            prompt_tokens: u32::MAX,
            completion_tokens: u32::MAX,
            ..Default::default()
        };
        usage.add(&request, &ModelPricing::default());
        usage.add(&request, &ModelPricing::default());
        assert_eq!(usage.input_tokens, 2 * u64::from(u32::MAX));
        assert_eq!(usage.output_tokens, 2 * u64::from(u32::MAX));
    }
}
