//! Boundless credit balance from `GET /v1/credits`.

use flashmind_types::{CreditBalance, CreditScope, CreditUnit};
use rust_decimal::Decimal;
use serde::Deserialize;

#[derive(Deserialize)]
pub(crate) struct Credits {
    balance: Option<Decimal>,
    total_credits: Option<Decimal>,
    total_spent: Option<Decimal>,
    key: Option<KeySpend>,
}

#[derive(Deserialize)]
struct KeySpend {
    spent: Option<Decimal>,
    limit: Option<Decimal>,
}

impl Credits {
    /// A key budget is the tighter limit, so it wins over the team balance.
    pub(crate) fn into_balance(self) -> CreditBalance {
        if let Some(KeySpend {
            spent,
            limit: Some(limit),
        }) = self.key
        {
            return CreditBalance {
                scope: CreditScope::ApiKey,
                unit: CreditUnit::Usd,
                remaining: spent.and_then(|spent| limit.checked_sub(spent)),
                total: Some(limit),
                used: spent,
                resets: None,
            };
        }
        CreditBalance {
            scope: CreditScope::Account,
            unit: CreditUnit::Usd,
            remaining: self.balance,
            total: self.total_credits,
            used: self.total_spent,
            resets: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn balance(value: serde_json::Value) -> CreditBalance {
        serde_json::from_value::<Credits>(value)
            .unwrap()
            .into_balance()
    }

    #[test]
    fn team_balance_without_a_key_budget() {
        let balance = balance(json!({
            "object": "credits", "currency": "usd",
            "balance": 18.5, "total_credits": 100, "total_spent": 81.5,
            "key": { "spent": 12.04, "limit": null }
        }));
        assert_eq!(balance.scope, CreditScope::Account);
        assert_eq!(balance.remaining, Some(Decimal::new(185, 1)));
        assert_eq!(balance.total, Some(Decimal::from(100)));
        assert_eq!(balance.used, Some(Decimal::new(815, 1)));
    }

    #[test]
    fn key_budget_takes_precedence() {
        let balance = balance(json!({
            "balance": 18.5, "total_credits": 100, "total_spent": 81.5,
            "key": { "spent": 2.5, "limit": 10 }
        }));
        assert_eq!(balance.scope, CreditScope::ApiKey);
        assert_eq!(balance.remaining, Some(Decimal::new(75, 1)));
        assert_eq!(balance.total, Some(Decimal::from(10)));
        assert_eq!(balance.used, Some(Decimal::new(25, 1)));
    }

    #[test]
    fn unlimited_team_reports_unknown_balance() {
        let balance = balance(json!({
            "balance": null, "total_credits": null, "total_spent": 3, "key": null
        }));
        assert_eq!(balance.scope, CreditScope::Account);
        assert_eq!(balance.remaining, None);
        assert_eq!(balance.total, None);
        assert_eq!(balance.used, Some(Decimal::from(3)));
    }
}
