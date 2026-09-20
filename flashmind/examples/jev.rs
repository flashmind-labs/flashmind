//! Classify a support ticket with Jev through OpenRouter.
//!
//! ```sh
//! cargo run -p flashmind --example jev -- --api-key YOUR_KEY
//! ```

use std::collections::BTreeMap;

use clap::Parser;

use flashmind::llm::{JEV_LATEST_MODEL, JevQuestion, JevRequest, OpenRouterProvider};

#[derive(Parser)]
#[command(about = "Evaluate typed decisions with Jev through OpenRouter")]
struct Args {
    #[arg(long, env = "OPENROUTER_API_KEY")]
    api_key: String,

    #[arg(long, default_value = JEV_LATEST_MODEL)]
    model: String,

    #[arg(
        long,
        default_value = "I was charged twice for my subscription and need a refund."
    )]
    ticket: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let provider = OpenRouterProvider::new(args.api_key);

    let request = JevRequest {
        model: args.model,
        state: serde_json::json!({"ticket": args.ticket}),
        questions: BTreeMap::from([
            (
                "refund".into(),
                JevQuestion::Noul {
                    instructions: "Is the customer asking for a refund?".into(),
                    criteria: None,
                },
            ),
            (
                "team".into(),
                JevQuestion::Choice {
                    instructions: "Which team should handle this ticket?".into(),
                    criteria: BTreeMap::from([
                        ("billing".into(), "Charges, refunds, and invoices".into()),
                        ("technical".into(), "Product defects and outages".into()),
                    ]),
                },
            ),
            (
                "urgency".into(),
                JevQuestion::Score {
                    instructions: "How urgent is this ticket?".into(),
                    criteria: vec![
                        "Can wait for the next release".into(),
                        "Should be resolved this week".into(),
                        "Blocking the customer now".into(),
                    ],
                },
            ),
        ]),
    };

    let response = provider.decide(request).await?;
    println!("model: {}", response.model);
    for (question, answer) in response.answers {
        println!("{question}: {answer}");
    }

    Ok(())
}
