//! Provider credit balances, separate from session token usage.

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// What the reported credit limit applies to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CreditScope {
    Account,
    ApiKey,
    Budget,
}

/// Unit used by a provider's credit balance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CreditUnit {
    Credits,
    Usd,
}

/// A provider balance snapshot. Missing amounts are unknown, never zero.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreditBalance {
    pub scope: CreditScope,
    pub unit: CreditUnit,
    #[serde(with = "rust_decimal::serde::str_option")]
    pub remaining: Option<Decimal>,
    #[serde(with = "rust_decimal::serde::str_option")]
    pub total: Option<Decimal>,
    #[serde(with = "rust_decimal::serde::str_option")]
    pub used: Option<Decimal>,
    pub resets: Option<String>,
}
