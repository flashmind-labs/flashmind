# flashmind-cron

Cron job scheduling with pluggable storage for [Flashmind](https://github.com/flashmind-labs/flashmind).

POSIX cron parsing, recurring/one-shot/on-wake schedules, TOML file backend.

## Usage

```rust
use flashmind_cron::{CronRegistry, CronRunner, TomlCronStore, JobSchedule};

let store = TomlCronStore::new("cron.toml");
let registry = CronRegistry::new(store);

// Add a recurring job
registry.add("cleanup", JobSchedule::Cron("0 3 * * *".into()), handler).await?;

// Run the scheduler
let runner = CronRunner::new(registry);
runner.run().await?;
```

## Schedule Types

- `Cron("0 */5 * * *")`  -  standard POSIX 5-field expressions
- `Once(datetime)`  -  fire once at a specific time
- `OnWake { from_hour, min_gap_secs, max_gap_secs }`  -  fire on wake within time window

## License

MPL-2.0
