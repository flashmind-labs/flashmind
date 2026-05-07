//! Cron schedule matching and next-fire-time computation.

use chrono::{DateTime, Datelike, TimeZone, Timelike, Utc};

use crate::error::CronError;
use crate::parse::{CronExpr, parse_cron};

// ---------------------------------------------------------------------------
// CronSchedule
// ---------------------------------------------------------------------------

/// A parsed cron schedule that can test whether a given time matches and
/// compute the next matching time after a given instant.
pub struct CronSchedule {
    expr: CronExpr,
}

impl CronSchedule {
    /// Parse a 5-field POSIX cron expression into a schedule.
    pub fn parse(s: &str) -> Result<Self, CronError> {
        Ok(Self {
            expr: parse_cron(s)?,
        })
    }

    /// Returns true if the given UTC datetime matches this cron schedule.
    ///
    /// POSIX semantics: when both day_of_month and day_of_week are restricted
    /// (not wildcards), the schedule fires if EITHER matches (OR logic).
    pub fn matches(&self, dt: &DateTime<Utc>) -> bool {
        if !self.expr.minute.contains(dt.minute()) {
            return false;
        }
        if !self.expr.hour.contains(dt.hour()) {
            return false;
        }
        if !self.expr.month.contains(dt.month()) {
            return false;
        }

        let dom_restricted = !self.expr.day_of_month.is_all();
        let dow_restricted = !self.expr.day_of_week.is_all();
        let dom_match = self.expr.day_of_month.contains(dt.day());
        let dow_match = self
            .expr
            .day_of_week
            .contains(dt.weekday().num_days_from_sunday());

        match (dom_restricted, dow_restricted) {
            (true, true) => dom_match || dow_match,
            _ => dom_match && dow_match,
        }
    }

    /// Compute the next UTC datetime after `after` that matches this schedule.
    ///
    /// Returns `None` if no match is found within 2 years (guards against
    /// impossible schedules like Feb 31).
    pub fn next_after(&self, after: &DateTime<Utc>) -> Option<DateTime<Utc>> {
        let limit = *after + chrono::Duration::days(366 * 2);
        let mut dt = *after + chrono::Duration::minutes(1);
        // Snap to start of minute
        dt = Utc
            .with_ymd_and_hms(dt.year(), dt.month(), dt.day(), dt.hour(), dt.minute(), 0)
            .single()?;

        while dt <= limit {
            // Skip months that can't match
            if !self.expr.month.contains(dt.month()) {
                dt = advance_month(dt)?;
                continue;
            }

            // Skip days that can't match
            if !self.day_matches(&dt) {
                dt = advance_day(dt)?;
                continue;
            }

            // Skip hours that can't match
            if !self.expr.hour.contains(dt.hour()) {
                dt = advance_hour(dt)?;
                continue;
            }

            if self.expr.minute.contains(dt.minute()) {
                return Some(dt);
            }

            // Find next matching minute in this hour
            if let Some(next_min) = next_set_value(&self.expr.minute, dt.minute() + 1) {
                dt = Utc
                    .with_ymd_and_hms(dt.year(), dt.month(), dt.day(), dt.hour(), next_min, 0)
                    .single()?;
                if dt <= limit {
                    return Some(dt);
                }
            }

            dt = advance_hour(dt)?;
        }

        None
    }

    fn day_matches(&self, dt: &DateTime<Utc>) -> bool {
        let dom_restricted = !self.expr.day_of_month.is_all();
        let dow_restricted = !self.expr.day_of_week.is_all();
        let dom_match = self.expr.day_of_month.contains(dt.day());
        let dow_match = self
            .expr
            .day_of_week
            .contains(dt.weekday().num_days_from_sunday());

        match (dom_restricted, dow_restricted) {
            (true, true) => dom_match || dow_match,
            _ => dom_match && dow_match,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn next_set_value(fs: &crate::parse::FieldSet, from: u32) -> Option<u32> {
    (from..=fs.max_val()).find(|&v| fs.contains(v))
}

fn advance_month(dt: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let (year, month) = if dt.month() == 12 {
        (dt.year() + 1, 1)
    } else {
        (dt.year(), dt.month() + 1)
    };
    Utc.with_ymd_and_hms(year, month, 1, 0, 0, 0).single()
}

fn advance_day(dt: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let next = dt + chrono::Duration::days(1);
    Utc.with_ymd_and_hms(next.year(), next.month(), next.day(), 0, 0, 0)
        .single()
}

fn advance_hour(dt: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let next = dt + chrono::Duration::hours(1);
    Utc.with_ymd_and_hms(next.year(), next.month(), next.day(), next.hour(), 0, 0)
        .single()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn utc(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, mo, d, h, mi, 0).unwrap()
    }

    #[test]
    fn test_matches_every_minute() {
        let sched = CronSchedule::parse("* * * * *").unwrap();
        assert!(sched.matches(&utc(2025, 6, 15, 10, 30)));
    }

    #[test]
    fn test_matches_specific() {
        let sched = CronSchedule::parse("30 14 * * *").unwrap();
        assert!(sched.matches(&utc(2025, 6, 15, 14, 30)));
        assert!(!sched.matches(&utc(2025, 6, 15, 14, 31)));
        assert!(!sched.matches(&utc(2025, 6, 15, 13, 30)));
    }

    #[test]
    fn test_matches_dom_dow_or_logic() {
        // Fire on the 1st OR on Mondays
        let sched = CronSchedule::parse("0 0 1 * MON").unwrap();
        // 2025-06-01 is a Sunday
        assert!(sched.matches(&utc(2025, 6, 1, 0, 0)));
        // 2025-06-02 is a Monday
        assert!(sched.matches(&utc(2025, 6, 2, 0, 0)));
        // 2025-06-03 is a Tuesday and not the 1st
        assert!(!sched.matches(&utc(2025, 6, 3, 0, 0)));
    }

    #[test]
    fn test_next_after_basic() {
        let sched = CronSchedule::parse("30 14 * * *").unwrap();
        let after = utc(2025, 6, 15, 14, 0);
        let next = sched.next_after(&after).unwrap();
        assert_eq!(next, utc(2025, 6, 15, 14, 30));
    }

    #[test]
    fn test_next_after_wraps_day() {
        let sched = CronSchedule::parse("0 9 * * *").unwrap();
        let after = utc(2025, 6, 15, 10, 0);
        let next = sched.next_after(&after).unwrap();
        assert_eq!(next, utc(2025, 6, 16, 9, 0));
    }

    #[test]
    fn test_next_after_wraps_month() {
        let sched = CronSchedule::parse("0 0 1 * *").unwrap();
        let after = utc(2025, 6, 2, 0, 0);
        let next = sched.next_after(&after).unwrap();
        assert_eq!(next, utc(2025, 7, 1, 0, 0));
    }

    #[test]
    fn test_next_after_step() {
        let sched = CronSchedule::parse("*/15 * * * *").unwrap();
        let after = utc(2025, 6, 15, 10, 1);
        let next = sched.next_after(&after).unwrap();
        assert_eq!(next, utc(2025, 6, 15, 10, 15));
    }

    #[test]
    fn test_next_after_returns_none_for_impossible() {
        // Feb 31 can never happen
        let sched = CronSchedule::parse("0 0 31 2 *").unwrap();
        let after = utc(2025, 1, 1, 0, 0);
        assert!(sched.next_after(&after).is_none());
    }

    #[test]
    fn test_next_after_skips_to_same_minute_plus_one() {
        let sched = CronSchedule::parse("30 14 * * *").unwrap();
        // If we're exactly at 14:30, next should be tomorrow 14:30
        let after = utc(2025, 6, 15, 14, 30);
        let next = sched.next_after(&after).unwrap();
        assert_eq!(next, utc(2025, 6, 16, 14, 30));
    }
}
