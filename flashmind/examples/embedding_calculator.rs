//! Add and subtract text embeddings, with parentheses and quoted phrases.
//!
//! ```sh
//! ollama pull nomic-embed-text
//! cargo run -p flashmind --example embedding_calculator --features memory -- 'king - queen'
//! cargo run -p flashmind --example embedding_calculator --features memory -- '(king + man) - (queen + woman)'
//! # Omit the expression for an interactive prompt. Use --full-vector for every component.
//! # Hosted alternative, with OPENROUTER_API_KEY set in the environment:
//! cargo run -p flashmind --example embedding_calculator --features memory -- --provider openrouter 'king - queen'
//! # OpenAI, with OPENAI_API_KEY set (or pass --api-key):
//! cargo run -p flashmind --example embedding_calculator --features memory -- --provider openai 'king - queen'
//! ```
//!
//! `--url` overrides the Ollama server or the full OpenAI-compatible embeddings
//! endpoint. API keys are optional for OpenAI-compatible endpoints.
//!
//! Subtraction returns a vector, not a similarity score. For a binary expression,
//! cosine similarity and Euclidean distance compare its outer operands. For
//! `(king + man) - (queen + woman)`, those are the two sums. Cosine ranges from
//! -1 to 1, with 1 meaning the same direction; it is undefined for zero vectors.
//! Results depend on the model and are not guaranteed to follow word analogies.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::io::{Write, stderr, stdin, stdout};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail, ensure};
use clap::Parser;
use nom::branch::alt;
use nom::bytes::complete::{take_until, take_while, take_while1};
use nom::character::complete::{char, one_of};
use nom::combinator::{all_consuming, cut, map, verify};
use nom::error::{Error, ErrorKind};
use nom::multi::many_m_n;
use nom::sequence::{preceded, terminated};
use nom::{Err as NomErr, IResult, Parser as _};
use secrecy::{ExposeSecret, SecretString};
use tokio::time::timeout;
use tracing_subscriber::EnvFilter;

use flashmind::memory::{EmbeddingProvider, EmbeddingProviderConfig, create_embedding_provider};
use flashmind::types::Provider;

const EMBEDDING_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Parser)]
#[command(about = "A toy calculator for text embeddings (+, -, and parentheses)")]
struct Args {
    /// Expression to evaluate; omit for an interactive prompt. Quote it in the shell.
    #[arg(allow_hyphen_values = true)]
    expression: Option<String>,

    /// Embedding backend: ollama, openai, or openrouter.
    #[arg(long, default_value = "ollama")]
    provider: Provider,

    /// Defaults to nomic-embed-text, text-embedding-3-small, or openai/text-embedding-3-small.
    #[arg(long, env = "EMBEDDING_MODEL")]
    model: Option<String>,

    /// Ollama server URL or full OpenAI-compatible embeddings endpoint.
    #[arg(long)]
    url: Option<String>,

    /// Optional bearer token; defaults to OPENAI_API_KEY or OPENROUTER_API_KEY.
    #[arg(long)]
    api_key: Option<SecretString>,

    /// Print every vector component instead of a preview.
    #[arg(long)]
    full_vector: bool,
}

impl Args {
    fn embedding_provider(&self) -> Result<Arc<dyn EmbeddingProvider>> {
        let model = |default: &str| self.model.clone().unwrap_or_else(|| default.to_owned());
        let key = |variable| {
            self.api_key
                .clone()
                .or_else(|| env::var(variable).ok().map(SecretString::from))
                .filter(|key| !key.expose_secret().is_empty())
                .map(|key| key.expose_secret().to_owned())
        };
        let config = match &self.provider {
            Provider::Ollama => {
                ensure!(
                    self.api_key.is_none(),
                    "Ollama does not support --api-key in this example"
                );
                EmbeddingProviderConfig::Ollama {
                    url: self.url.clone().or_else(|| env::var("OLLAMA_URL").ok()),
                    model: model("nomic-embed-text"),
                }
            }
            Provider::OpenAi => EmbeddingProviderConfig::OpenAI {
                api_key: key("OPENAI_API_KEY"),
                model: model("text-embedding-3-small"),
                base_url: self
                    .url
                    .clone()
                    .or_else(|| env::var("OPENAI_EMBEDDINGS_URL").ok()),
            },
            Provider::OpenRouter => {
                ensure!(
                    self.url.is_none(),
                    "--url is only supported with Ollama or OpenAI"
                );
                EmbeddingProviderConfig::OpenRouter {
                    api_key: key("OPENROUTER_API_KEY"),
                    model: model("openai/text-embedding-3-small"),
                }
            }
            provider => {
                bail!("No embedding backend for {provider}; use ollama, openai, or openrouter")
            }
        };
        Ok(create_embedding_provider(&config, None)?)
    }
}

#[derive(Clone, Copy)]
enum Operation {
    Add,
    Subtract,
}

enum Expression<'a> {
    Text(&'a str),
    Negate(Box<Self>),
    Binary {
        operation: Operation,
        left: Box<Self>,
        right: Box<Self>,
    },
}

impl<'a> Expression<'a> {
    fn parse(input: &'a str) -> Result<Self> {
        ensure!(input.chars().count() <= 4096, "Expression is too long");
        all_consuming(terminated(|input| parse_sum(input, 0), whitespace))
            .parse(input)
            .map(|(_, expression)| expression)
            .map_err(|error| anyhow!("Invalid expression: {error}"))
    }

    fn texts(&self, texts: &mut BTreeSet<&'a str>) {
        match self {
            Self::Text(text) => {
                texts.insert(*text);
            }
            Self::Negate(inner) => inner.texts(texts),
            Self::Binary { left, right, .. } => {
                left.texts(texts);
                right.texts(texts);
            }
        }
    }

    fn evaluate(&self, embeddings: &BTreeMap<String, Vec<f64>>) -> Result<Vec<f64>> {
        match self {
            Self::Text(text) => embeddings
                .get(*text)
                .cloned()
                .with_context(|| format!("Missing embedding for {text:?}")),
            Self::Negate(inner) => Ok(inner.evaluate(embeddings)?.iter().map(|x| -x).collect()),
            Self::Binary {
                operation,
                left,
                right,
            } => {
                let mut left = left.evaluate(embeddings)?;
                let right = right.evaluate(embeddings)?;
                ensure!(left.len() == right.len(), "Embedding dimensions differ");
                for (a, b) in left.iter_mut().zip(right) {
                    match operation {
                        Operation::Add => *a += b,
                        Operation::Subtract => *a -= b,
                    }
                }
                Ok(left)
            }
        }
    }
}

fn whitespace(input: &str) -> IResult<&str, &str> {
    take_while(char::is_whitespace).parse(input)
}

fn parse_sum(input: &str, depth: usize) -> IResult<&str, Expression<'_>> {
    let (input, first) = parse_term(input, depth)?;
    let (input, rest) = many_m_n(
        0,
        128,
        (
            preceded(whitespace, one_of("+-")),
            cut(|input| parse_term(input, depth)),
        ),
    )
    .parse(input)?;
    let expression = rest
        .into_iter()
        .fold(first, |left, (operator, right)| Expression::Binary {
            operation: if operator == '+' {
                Operation::Add
            } else {
                Operation::Subtract
            },
            left: Box::new(left),
            right: Box::new(right),
        });
    Ok((input, expression))
}

fn parse_term(input: &str, depth: usize) -> IResult<&str, Expression<'_>> {
    if depth >= 32 {
        return Err(NomErr::Failure(Error::new(input, ErrorKind::TooLarge)));
    }
    preceded(
        whitespace,
        alt((
            preceded(
                char('('),
                cut(terminated(
                    |input| parse_sum(input, depth + 1),
                    preceded(whitespace, char(')')),
                )),
            ),
            map(
                preceded(char('-'), cut(|input| parse_term(input, depth + 1))),
                |inner| Expression::Negate(Box::new(inner)),
            ),
            preceded(char('+'), cut(|input| parse_term(input, depth + 1))),
            map(
                alt((
                    preceded(
                        char('"'),
                        cut(terminated(
                            verify(take_until("\""), |text: &str| !text.trim().is_empty()),
                            char('"'),
                        )),
                    ),
                    take_while1(|c: char| {
                        !c.is_whitespace() && !matches!(c, '+' | '-' | '(' | ')' | '"')
                    }),
                )),
                Expression::Text,
            ),
        )),
    )
    .parse(input)
}

async fn load_embeddings(
    expression: &Expression<'_>,
    provider: &dyn EmbeddingProvider,
    cache: &mut BTreeMap<String, Vec<f64>>,
) -> Result<()> {
    let mut texts = BTreeSet::new();
    expression.texts(&mut texts);
    let missing: Vec<_> = texts
        .into_iter()
        .filter(|text| !cache.contains_key(*text))
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    eprintln!(
        "Fetching {} embeddings from {}...",
        missing.len(),
        provider.name()
    );
    let vectors = timeout(EMBEDDING_TIMEOUT, provider.embed_batch(&missing))
        .await
        .context(
            "Embedding request timed out after 60 seconds; check the provider and network",
        )??;
    ensure!(vectors.len() == missing.len(), "Unexpected embedding count");
    let dimensions = cache
        .values()
        .next()
        .map(Vec::len)
        .unwrap_or_else(|| vectors[0].len());
    ensure!(dimensions > 0, "Received an empty embedding");
    for vector in &vectors {
        ensure!(vector.len() == dimensions, "Embedding dimensions differ");
        ensure!(vector.iter().all(|x| x.is_finite()), "Non-finite embedding");
    }
    for (text, vector) in missing.into_iter().zip(vectors) {
        cache.insert(text.to_owned(), vector.into_iter().map(f64::from).collect());
    }
    Ok(())
}

fn norm(vector: &[f64]) -> f64 {
    vector.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn cosine(left: &[f64], right: &[f64]) -> Option<f64> {
    let denominator = norm(left) * norm(right);
    (denominator > 0.0).then(|| {
        (left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>() / denominator).clamp(-1.0, 1.0)
    })
}

async fn calculate(
    input: &str,
    provider: &dyn EmbeddingProvider,
    cache: &mut BTreeMap<String, Vec<f64>>,
    full_vector: bool,
) -> Result<()> {
    let expression = Expression::parse(input)?;
    load_embeddings(&expression, provider, cache).await?;
    let result = expression.evaluate(cache)?;
    println!("Dimensions: {}", result.len());
    println!("Result L2 norm: {:.6}", norm(&result));
    if let Expression::Binary { left, right, .. } = &expression {
        let left = left.evaluate(cache)?;
        let right = right.evaluate(cache)?;
        match cosine(&left, &right) {
            Some(score) => println!("Cosine similarity between outer operands: {score:.6}"),
            None => println!("Cosine similarity: undefined (a zero vector operand)"),
        }
        let distance = left
            .iter()
            .zip(&right)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        println!("Euclidean distance between outer operands: {distance:.6}");
    }
    let shown = if full_vector {
        result.len()
    } else {
        result.len().min(8)
    };
    println!(
        "Result vector ({shown}/{} components): {:?}",
        result.len(),
        &result[..shown]
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()))
        .with_writer(stderr)
        .init();
    let provider = args.embedding_provider()?;
    let mut cache = BTreeMap::new();
    if let Some(input) = args.expression {
        return calculate(&input, provider.as_ref(), &mut cache, args.full_vector).await;
    }

    println!("Embedding calculator: +, -, parentheses, and double-quoted phrases.");
    println!("Try (king + man) - (queen + woman). Type quit to exit.");
    loop {
        print!("> ");
        stdout().flush()?;
        let mut input = String::new();
        if stdin().read_line(&mut input)? == 0 || input.trim() == "quit" {
            break;
        }
        if input.trim().is_empty() {
            continue;
        }
        if let Err(error) = calculate(&input, provider.as_ref(), &mut cache, args.full_vector).await
        {
            eprintln!("Error: {error:#}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use flashmind::memory::{FlashmemError, Result as EmbeddingResult};

    use super::*;

    #[derive(Default)]
    struct BatchEmbedder {
        vectors: Vec<Vec<f32>>,
        calls: AtomicUsize,
        delay: Duration,
    }

    #[async_trait]
    impl EmbeddingProvider for BatchEmbedder {
        async fn embed(&self, _text: &str) -> EmbeddingResult<Vec<f32>> {
            Err(FlashmemError::Memory("Use the batch endpoint".into()))
        }

        async fn embed_batch(&self, _texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            tokio::time::sleep(self.delay).await;
            Ok(self.vectors.clone())
        }

        fn dimensions(&self) -> usize {
            2
        }

        fn name(&self) -> &str {
            "test"
        }
    }

    fn evaluate(input: &str) -> Vec<f64> {
        let embeddings = BTreeMap::from([
            ("king".into(), vec![3.0, 2.0]),
            ("queen".into(), vec![3.0, 1.0]),
            ("man".into(), vec![0.0, 2.0]),
            ("woman".into(), vec![0.0, 1.0]),
            ("ice cream".into(), vec![1.0, 4.0]),
        ]);
        Expression::parse(input)
            .unwrap()
            .evaluate(&embeddings)
            .unwrap()
    }

    #[test]
    fn arithmetic_and_grouping() {
        assert_eq!(evaluate("king - queen"), [0.0, 1.0]);
        assert_eq!(evaluate("(king + man) - (queen + woman)"), [0.0, 2.0]);
        assert_eq!(evaluate("king - (queen - man)"), [0.0, 3.0]);
        assert_eq!(evaluate("king - queen - man"), [0.0, -1.0]);
        assert_eq!(evaluate("-(king - queen) + +man"), [0.0, 1.0]);
        assert_eq!(evaluate("\"ice cream\" - woman"), [1.0, 3.0]);
        assert_eq!(evaluate("king - king"), [0.0, 0.0]);
        assert_eq!(evaluate("\u{2003}king\u{a0}-queen\n"), [0.0, 1.0]);
    }

    #[test]
    fn malformed_expressions_are_rejected() {
        for input in [
            "",
            "()",
            "king +",
            "(king",
            "king)",
            "king queen",
            "king * man",
            "\"\"",
            "\"   \"",
            "\"king",
            "king + ()",
            "king + (queen",
            "king + \"queen",
        ] {
            assert!(Expression::parse(input).is_err(), "Accepted {input:?}");
        }
        assert!(Expression::parse(&format!("{}king{}", "(".repeat(40), ")".repeat(40))).is_err());
        assert!(Expression::parse(&"k".repeat(4097)).is_err());
        assert!(Expression::parse(&format!("{}king", "king + ".repeat(128))).is_ok());
        assert!(Expression::parse(&format!("{}king", "king + ".repeat(129))).is_err());
    }

    #[test]
    fn openai_accepts_optional_keys_and_a_custom_endpoint() {
        for key in [None, Some(""), Some("example-key")] {
            let mut arguments = vec![
                "calculator",
                "--provider",
                "openai",
                "--url",
                "https://example.com/v1/embeddings",
            ];
            if let Some(key) = key {
                arguments.extend(["--api-key", key]);
            }
            let args = Args::try_parse_from(arguments).unwrap();
            assert_eq!(args.provider, Provider::OpenAi);
            assert_eq!(args.embedding_provider().unwrap().name(), "openai");
        }
        assert!(!format!("{:?}", SecretString::from("example-key")).contains("example-key"));
        let unsupported = Args::try_parse_from(["calculator", "--provider", "anthropic"]).unwrap();
        assert!(unsupported.embedding_provider().is_err());
    }

    #[test]
    fn similarity_handles_direction_and_zero_vectors() {
        assert_eq!(cosine(&[1.0, 0.0], &[1.0, 0.0]), Some(1.0));
        assert_eq!(cosine(&[1.0, 0.0], &[0.0, 1.0]), Some(0.0));
        assert_eq!(cosine(&[1.0, 0.0], &[-1.0, 0.0]), Some(-1.0));
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), None);
    }

    #[tokio::test]
    async fn repeated_terms_are_batched_once_and_cached() {
        let provider = BatchEmbedder {
            vectors: vec![vec![3.0, 2.0], vec![3.0, 1.0]],
            ..Default::default()
        };
        let expression = Expression::parse("king - queen + king").unwrap();
        let mut cache = BTreeMap::new();
        load_embeddings(&expression, &provider, &mut cache)
            .await
            .unwrap();
        assert_eq!(expression.evaluate(&cache).unwrap(), [3.0, 3.0]);
        load_embeddings(&expression, &provider, &mut cache)
            .await
            .unwrap();
        assert_eq!(provider.calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn invalid_batches_do_not_enter_the_cache() {
        let expression = Expression::parse("king - queen").unwrap();
        for vectors in [
            vec![],
            vec![vec![1.0, 2.0]],
            vec![vec![], vec![]],
            vec![vec![1.0, 2.0], vec![1.0]],
            vec![vec![1.0, 2.0], vec![f32::NAN, 0.0]],
        ] {
            let provider = BatchEmbedder {
                vectors,
                ..Default::default()
            };
            let mut cache = BTreeMap::new();
            assert!(
                load_embeddings(&expression, &provider, &mut cache)
                    .await
                    .is_err()
            );
            assert!(cache.is_empty());
        }
    }

    #[tokio::test(start_paused = true)]
    async fn stalled_requests_time_out_without_changing_the_cache() {
        let provider = BatchEmbedder {
            delay: EMBEDDING_TIMEOUT + Duration::from_secs(1),
            ..Default::default()
        };
        let expression = Expression::parse("man - woman").unwrap();
        let mut cache = BTreeMap::new();
        let error = load_embeddings(&expression, &provider, &mut cache)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("timed out after 60 seconds"));
        assert!(cache.is_empty());
    }
}
