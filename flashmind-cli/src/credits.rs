use std::fmt;
use std::time::Duration;

use flashmind_types::{CreditBalance, CreditScope, CreditUnit, LlmProvider};
use rust_decimal::Decimal;

pub(crate) struct CreditSummary<'a> {
    pub provider: &'a str,
    pub balance: &'a CreditBalance,
    pub consumed: Option<Decimal>,
}

impl fmt::Display for CreditSummary<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let amount = |value: Option<Decimal>| match value {
            Some(value) => match self.balance.unit {
                CreditUnit::Usd => format!("${}", value.normalize()),
                CreditUnit::Credits => value.normalize().to_string(),
            },
            None => "unavailable".to_owned(),
        };
        let scope = match self.balance.scope {
            CreditScope::Account => "account",
            CreditScope::ApiKey => "API key limit",
            CreditScope::Budget => "budget",
        };
        write!(f, "{} credits ({scope}): ", self.provider)?;
        if self.balance.unit == CreditUnit::Credits || self.consumed.is_some() {
            write!(f, "{} used this call, ", amount(self.consumed))?;
        }
        write!(
            f,
            "{} remaining, {} total",
            amount(self.balance.remaining),
            amount(self.balance.total)
        )?;
        if let Some(reset) = &self.balance.resets {
            write!(f, " (resets {reset})")?;
        }
        if self.provider == "openrouter" && self.balance.remaining.is_none() {
            write!(f, " (account balance requires a management key)")?;
        }
        Ok(())
    }
}

/// Balance reporting is optional and cannot turn successful generation into an error.
pub(crate) async fn fetch(provider: &dyn LlmProvider) -> Option<CreditBalance> {
    match tokio::time::timeout(Duration::from_secs(6), provider.credit_balance()).await {
        Ok(Ok(balance)) => balance,
        Ok(Err(_)) | Err(_) => {
            tracing::warn!(provider = provider.name(), "credit balance unavailable");
            None
        }
    }
}

pub(crate) async fn report(provider: &dyn LlmProvider) {
    if let Some(balance) = fetch(provider).await {
        eprintln!(
            "{}",
            CreditSummary {
                provider: provider.name(),
                balance: &balance,
                consumed: None
            }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_total_and_zero_remaining_are_distinct() {
        let balance = CreditBalance {
            scope: CreditScope::Account,
            unit: CreditUnit::Credits,
            remaining: Some(Decimal::ZERO),
            total: None,
            used: None,
            resets: None,
        };
        assert_eq!(
            CreditSummary {
                provider: "meshy",
                balance: &balance,
                consumed: Some(Decimal::from(30)),
            }
            .to_string(),
            "meshy credits (account): 30 used this call, 0 remaining, unavailable total"
        );
        let mut unknown = balance;
        unknown.remaining = None;
        assert!(
            CreditSummary {
                provider: "meshy",
                balance: &unknown,
                consumed: None,
            }
            .to_string()
            .contains("unavailable remaining")
        );
    }

    #[test]
    fn api_key_balances_retain_precision_and_reset_period() {
        let balance = CreditBalance {
            scope: CreditScope::ApiKey,
            unit: CreditUnit::Usd,
            remaining: Some(Decimal::new(1234567, 6)),
            total: Some(Decimal::TEN),
            used: None,
            resets: Some("monthly".into()),
        };
        assert_eq!(
            CreditSummary {
                provider: "openrouter",
                balance: &balance,
                consumed: None,
            }
            .to_string(),
            "openrouter credits (API key limit): $1.234567 remaining, $10 total (resets monthly)"
        );
        let encoded = serde_json::to_value(&balance).unwrap();
        assert_eq!(encoded["remaining"], "1.234567");
    }
}
