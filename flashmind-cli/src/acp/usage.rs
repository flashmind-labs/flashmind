//! A session's context use and cost, reported to the client as
//! `usage_update`.

use agent_client_protocol::v1::schema::{Cost, SessionUpdate, UsageUpdate};
use flashmind_types::{AgentEvent, Model, ModelPricing, SessionUsage, TokenUsage};
use rust_decimal::prelude::ToPrimitive;
use std::collections::BTreeMap;

/// Adds up what a session's model calls cost and remembers how much context
/// the last one used.
pub(crate) struct Meter {
    model: Model,
    pricing: ModelPricing,
    total: SessionUsage,
    pub(super) unsaved: BTreeMap<String, SessionUsage>,
    last: Option<TokenUsage>,
}

impl Meter {
    /// A meter for a new session on `model`, priced at `pricing`.
    pub(crate) fn new(model: Model, pricing: ModelPricing) -> Self {
        Self {
            model,
            pricing,
            total: SessionUsage::default(),
            last: None,
            unsaved: BTreeMap::new(),
        }
    }

    /// Goes on from the totals and the last call saved with a session.
    pub(crate) fn resumed(mut self, total: SessionUsage, last: Option<TokenUsage>) -> Self {
        self.total = total;
        self.last = last;
        self
    }

    /// Changes the price for future calls while keeping the session's totals.
    pub(super) fn switch_model(&mut self, model: Model, pricing: ModelPricing) {
        self.model = model;
        self.pricing = pricing;
    }

    /// Counts the model call `event` reports, if it reports one. Calls of
    /// delegated agents add to the cost but not to the session's context; a
    /// delegated call without a billed charge can use the parent's prices
    /// only when it ran on the same model.
    pub(super) fn record(&mut self, event: &AgentEvent) -> bool {
        let mut model = &self.model;
        let mut current = event;
        while let AgentEvent::SpawnedEvent {
            model: child,
            event,
            ..
        } = current
        {
            model = child.as_ref().unwrap_or(model);
            current = event;
        }
        let usage = match current {
            AgentEvent::Usage(usage) => usage,
            AgentEvent::CompactionUsage {
                model: compact,
                usage,
            } => {
                model = compact;
                usage
            }
            _ => return false,
        };
        if matches!(event, AgentEvent::Usage(_)) {
            self.last = Some(usage.clone());
        }
        let unknown = ModelPricing::default();
        let pricing = if model == &self.model {
            &self.pricing
        } else {
            &unknown
        };
        self.total.add(usage, pricing);
        let billed_model = usage
            .response_model
            .as_deref()
            .filter(|name| !name.is_empty())
            .map_or_else(
                || model.to_string(),
                |name| format!("{}:{name}", model.provider),
            );
        self.unsaved
            .entry(billed_model)
            .or_default()
            .add(usage, pricing);
        true
    }

    /// The `usage_update` for a context window of `size` tokens: the context
    /// the last call left (its prompt and its reply) and the session's cost
    /// so far, which is left out while any call's price is unknown. `None`
    /// before the first call.
    pub(super) fn update(&self, size: u32) -> Option<SessionUpdate> {
        let last = self.last.as_ref()?;
        let used = u64::from(last.prompt_tokens) + u64::from(last.completion_tokens);
        let mut update = UsageUpdate::new(used, u64::from(size));
        if !self.total.cost_unknown
            && let Some(amount) = self.total.total_cost.to_f64()
        {
            update = update.cost(Cost::new(amount, "USD"));
        }
        update.meta = Some(serde_json::Map::from_iter([(
            "flsh/sessionUsage".to_owned(),
            serde_json::json!(self.total),
        )]));
        Some(SessionUpdate::UsageUpdate(update))
    }

    /// The session's totals, saved with it.
    pub(super) fn total(&self) -> &SessionUsage {
        &self.total
    }

    /// The last call of the session's own agent, saved with it.
    pub(super) fn last(&self) -> Option<&TokenUsage> {
        self.last.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use rust_decimal::Decimal;

    use super::*;

    fn model() -> Model {
        "ollama:script".parse().expect("the model id is valid")
    }

    fn priced() -> ModelPricing {
        ModelPricing {
            prompt: Some(Decimal::new(2, 6)),
            completion: Some(Decimal::new(10, 6)),
            ..Default::default()
        }
    }

    fn call(prompt_tokens: u32, completion_tokens: u32) -> AgentEvent {
        AgentEvent::Usage(TokenUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            ..Default::default()
        })
    }

    fn usage_update(meter: &Meter, size: u32) -> UsageUpdate {
        match meter.update(size) {
            Some(SessionUpdate::UsageUpdate(update)) => update,
            other => panic!("expected a usage update, got {other:?}"),
        }
    }

    #[test]
    fn nothing_is_reported_before_the_first_call() {
        let mut meter = Meter::new(model(), priced());
        assert!(!meter.record(&AgentEvent::TextDelta("hi".into())));
        assert!(meter.update(1_000).is_none());
    }

    #[test]
    fn a_model_switch_keeps_past_costs_and_prices_future_calls() {
        let mut meter = Meter::new(model(), priced());
        meter.record(&call(1_000, 100));
        meter.switch_model(
            "ollama:other".parse().unwrap(),
            ModelPricing {
                prompt: Some(Decimal::new(4, 6)),
                completion: Some(Decimal::new(20, 6)),
                ..Default::default()
            },
        );
        meter.record(&call(1_000, 100));
        let update = usage_update(&meter, 8_000);
        assert_eq!(update.used, 1_100);
        assert!((update.cost.unwrap().amount - 0.009).abs() < 1e-12);
    }

    #[test]
    fn routed_and_delegated_requests_keep_their_provider_charges_and_models() {
        let mut meter = Meter::new(model(), priced());
        let usage = |usd, name: &str| {
            AgentEvent::Usage(TokenUsage {
                prompt_tokens: 100,
                billed_usd: Some(Decimal::new(usd, 4)),
                response_model: Some(name.to_owned()),
                ..TokenUsage::default()
            })
        };
        meter.record(&usage(3, "served-model"));
        meter.record(&AgentEvent::SpawnedEvent {
            id: "worker".into(),
            task: "work".into(),
            model: Some("openrouter:requested-model".parse().unwrap()),
            event: Box::new(usage(17, "another-model")),
        });

        assert_eq!(meter.total().total_cost, Decimal::new(20, 4));
        assert_eq!(
            meter.unsaved["openrouter:another-model"].total_cost,
            Decimal::new(17, 4)
        );
        assert_eq!(
            meter.unsaved["ollama:served-model"].total_cost,
            Decimal::new(3, 4)
        );
        assert!(!meter.total().cost_unknown);
    }

    #[test]
    fn a_delegated_agent_on_another_model_leaves_the_cost_unknown() {
        let mut meter = Meter::new(model(), priced());
        meter.record(&call(1_000, 100));
        let same = AgentEvent::SpawnedEvent {
            id: "worker".into(),
            task: "work".into(),
            model: Some(model()),
            event: Box::new(call(500, 50)),
        };
        assert!(meter.record(&same));
        let update = usage_update(&meter, 8_000);
        // The delegated call adds to the cost, not to the context.
        assert_eq!(update.used, 1_100);
        let cost = update.cost.expect("both calls have a price");
        assert!((cost.amount - 0.0045).abs() < 1e-12, "{}", cost.amount);

        let other = AgentEvent::SpawnedEvent {
            id: "worker".into(),
            task: "work".into(),
            model: Some("ollama:other".parse().expect("the model id is valid")),
            event: Box::new(call(500, 50)),
        };
        assert!(meter.record(&other));
        assert!(usage_update(&meter, 8_000).cost.is_none());
    }
}
